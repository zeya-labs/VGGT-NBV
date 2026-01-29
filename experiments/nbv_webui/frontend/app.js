import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/OBJLoader.js";

const canvas = document.getElementById("three-canvas");
const modelInput = document.getElementById("model-path");
const loadButton = document.getElementById("load-model");
const statusEl = document.getElementById("model-status");
const addCameraButton = document.getElementById("add-camera");
const clearCameraButton = document.getElementById("clear-cameras");
const cameraListEl = document.getElementById("camera-list");
const calculateButton = document.getElementById("calculate-btn");
const historyEl = document.getElementById("history");
const clearHistoryButton = document.getElementById("clear-history");
const imageSizeInput = document.getElementById("image-size");
const fovInput = document.getElementById("fov");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e14);

const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
camera.position.set(0, 0, 3);

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
});
renderer.setPixelRatio(window.devicePixelRatio);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(5, 8, 4);
scene.add(dirLight);

let currentObject = null;
let currentMeshPath = null;
let cameraList = [];

function setStatus(message) {
  statusEl.textContent = message;
}

function resizeRenderer() {
  const { clientWidth, clientHeight } = canvas;
  if (clientWidth === 0 || clientHeight === 0) {
    return;
  }
  renderer.setSize(clientWidth, clientHeight, false);
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
}

window.addEventListener("resize", () => {
  resizeRenderer();
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function disposeObject(object) {
  object.traverse((child) => {
    if (child.geometry) {
      child.geometry.dispose();
    }
    if (child.material) {
      if (Array.isArray(child.material)) {
        child.material.forEach((mat) => mat.dispose());
      } else {
        child.material.dispose();
      }
    }
  });
}

function fitCameraToObject(object) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const distance = maxDim === 0 ? 3 : maxDim / Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));

  const direction = new THREE.Vector3(1, 1, 1).normalize();
  camera.position.copy(center.clone().add(direction.multiplyScalar(distance)));
  camera.near = Math.max(distance / 100, 0.001);
  camera.far = distance * 100;
  camera.updateProjectionMatrix();

  controls.target.copy(center);
  controls.update();
}

function renderCameraList() {
  cameraListEl.innerHTML = "";
  if (cameraList.length === 0) {
    cameraListEl.innerHTML = "<div class=\"status\">No cameras yet.</div>";
    return;
  }

  cameraList.forEach((cam, index) => {
    const item = document.createElement("div");
    item.className = "list-item";

    const pos = cam.position.map((v) => v.toFixed(3)).join(", ");
    const tgt = cam.target.map((v) => v.toFixed(3)).join(", ");

    const text = document.createElement("span");
    text.textContent = `#${index + 1} pos(${pos}) target(${tgt})`;

    const remove = document.createElement("button");
    remove.textContent = "Remove";
    remove.addEventListener("click", () => {
      cameraList.splice(index, 1);
      renderCameraList();
    });

    item.appendChild(text);
    item.appendChild(remove);
    cameraListEl.appendChild(item);
  });
}

function renderHistory(records) {
  historyEl.innerHTML = "";
  if (!records || records.length === 0) {
    historyEl.innerHTML = "<div class=\"status\">No history yet.</div>";
    return;
  }

  records
    .slice()
    .reverse()
    .forEach((record) => {
      const item = document.createElement("div");
      item.className = "history-item";

      const title = document.createElement("h3");
      title.textContent = `Run ${record.record_id}`;

      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = `Loss: ${record.loss_chamfer.toFixed(6)} | Cameras: ${record.cameras.length} | ${record.created_at}`;

      const images = document.createElement("div");
      images.className = "history-images";

      record.views.forEach((view, idx) => {
        const rgb = document.createElement("img");
        rgb.src = view.rgb;
        rgb.alt = `RGB ${idx + 1}`;

        const pts = document.createElement("img");
        pts.src = view.points;
        pts.alt = `Points ${idx + 1}`;

        images.appendChild(rgb);
        images.appendChild(pts);
      });

      item.appendChild(title);
      item.appendChild(meta);
      item.appendChild(images);
      historyEl.appendChild(item);
    });
}

async function fetchHistory() {
  const response = await fetch("/api/history");
  if (!response.ok) {
    return [];
  }
  return response.json();
}

async function clearHistory() {
  const response = await fetch("/api/history/clear", { method: "POST" });
  if (!response.ok) {
    setStatus(`Clear failed: ${await response.text()}`);
    return;
  }
  const payload = await response.json();
  renderHistory([]);
  const count = payload.deleted_runs ?? 0;
  setStatus(`History cleared (${count} runs).`);
}

async function loadMesh() {
  const meshPath = modelInput.value.trim();
  if (!meshPath) {
    setStatus("Please enter a model path.");
    return;
  }

  setStatus("Loading mesh info...");

  const infoResponse = await fetch(`/api/mesh_info?path=${encodeURIComponent(meshPath)}`);
  if (!infoResponse.ok) {
    setStatus(`Failed to load mesh info: ${await infoResponse.text()}`);
    return;
  }
  const info = await infoResponse.json();

  const textResponse = await fetch(`/api/mesh_text?path=${encodeURIComponent(meshPath)}`);
  if (!textResponse.ok) {
    setStatus(`Failed to load OBJ: ${await textResponse.text()}`);
    return;
  }
  const objText = await textResponse.text();

  const loader = new OBJLoader();
  const object = loader.parse(objText);
  object.traverse((child) => {
    if (child.isMesh) {
      child.material = new THREE.MeshStandardMaterial({
        color: 0xd1d5db,
        metalness: 0.1,
        roughness: 0.8,
      });
    }
  });

  if (currentObject) {
    scene.remove(currentObject);
    disposeObject(currentObject);
  }

  const scale = info.scale > 0 ? 1 / info.scale : 1.0;
  object.scale.setScalar(scale);
  object.position.set(
    -info.centroid[0] * scale,
    -info.centroid[1] * scale,
    -info.centroid[2] * scale
  );

  scene.add(object);
  currentObject = object;
  currentMeshPath = meshPath;

  fitCameraToObject(object);
  resizeRenderer();

  cameraList = [];
  renderCameraList();

  setStatus("Model loaded and normalized (quantile).");
}

function addCamera() {
  if (!currentObject) {
    setStatus("Load a model first.");
    return;
  }

  const pos = camera.position.clone();
  const tgt = controls.target.clone();

  cameraList.push({
    position: [pos.x, pos.y, pos.z],
    target: [tgt.x, tgt.y, tgt.z],
  });

  renderCameraList();
}

async function calculate() {
  if (!currentMeshPath) {
    setStatus("Load a model before calculating.");
    return;
  }
  if (cameraList.length === 0) {
    setStatus("Add at least one camera.");
    return;
  }

  const imageSize = Number.parseInt(imageSizeInput.value, 10) || 512;
  const fov = Number.parseFloat(fovInput.value) || 60;

  setStatus("Calculating... this may take a while.");

  const response = await fetch("/api/calculate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      mesh_path: currentMeshPath,
      cameras: cameraList,
      image_size: imageSize,
      fov,
    }),
  });

  if (!response.ok) {
    setStatus(`Calculation failed: ${await response.text()}`);
    return;
  }

  const record = await response.json();
  const history = await fetchHistory();
  renderHistory(history);

  setStatus(`Done. Chamfer: ${record.loss_chamfer.toFixed(6)}`);
}

loadButton.addEventListener("click", () => {
  loadMesh().catch((err) => setStatus(`Load error: ${err}`));
});

addCameraButton.addEventListener("click", addCamera);
clearCameraButton.addEventListener("click", () => {
  cameraList = [];
  renderCameraList();
});
calculateButton.addEventListener("click", () => {
  calculate().catch((err) => setStatus(`Error: ${err}`));
});
clearHistoryButton.addEventListener("click", () => {
  clearHistory().catch((err) => setStatus(`Error: ${err}`));
});

resizeRenderer();
animate();

fetchHistory().then(renderHistory).catch(() => {
  renderHistory([]);
});
