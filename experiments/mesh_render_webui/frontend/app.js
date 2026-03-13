import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/OBJLoader.js";
import { PLYLoader } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/PLYLoader.js";

const DEFAULT_MESH =
  "models/House3K_obj/BATCH_1/Set_A/BAT1_SETA_HOUSE1.obj";

const meshSearchInput = document.getElementById("mesh-search");
const meshList = document.getElementById("mesh-list");
const refreshMeshesButton = document.getElementById("refresh-meshes");
const loadMeshButton = document.getElementById("load-mesh");
const resetViewButton = document.getElementById("reset-view");
const meshStatus = document.getElementById("mesh-status");
const renderStatus = document.getElementById("render-status");
const captureButton = document.getElementById("capture-image");
const renderVideoButton = document.getElementById("render-video");
const clearHistoryButton = document.getElementById("clear-history");
const historyContainer = document.getElementById("history");
const imageSizeInput = document.getElementById("image-size");
const fovInput = document.getElementById("fov");
const durationInput = document.getElementById("duration-sec");
const fpsInput = document.getElementById("fps");
const trajectoryModeInput = document.getElementById("trajectory-mode");
const viewerTitle = document.getElementById("viewer-title");
const viewerBadge = document.getElementById("viewer-badge");
const viewerOverlay = document.getElementById("viewer-overlay");
const cameraReadout = document.getElementById("camera-readout");
const canvas = document.getElementById("viewer-canvas");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x071014);
scene.fog = new THREE.FogExp2(0x071014, 0.03);

const camera = new THREE.PerspectiveCamera(
  Number(fovInput.value),
  1,
  0.01,
  2000
);
camera.position.set(1.8, 1.4, 1.8);

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false,
});
renderer.setPixelRatio(window.devicePixelRatio);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 1.1));
const rimLight = new THREE.DirectionalLight(0xd8f8ff, 1.2);
rimLight.position.set(4, 7, 5);
scene.add(rimLight);

const fillLight = new THREE.DirectionalLight(0xb9ffdb, 0.6);
fillLight.position.set(-3, 1.5, -4);
scene.add(fillLight);

const grid = new THREE.GridHelper(6, 12, 0x6ad6a7, 0x2d4d49);
grid.material.opacity = 0.18;
grid.material.transparent = true;
grid.position.y = -1.2;
scene.add(grid);

let meshCatalog = [];
let currentObject = null;
let currentMeshPath = null;
let currentMeshInfo = null;
let currentHistory = [];
let isBusy = false;

function setMeshStatus(message) {
  meshStatus.textContent = message;
}

function setRenderStatus(message) {
  renderStatus.textContent = message;
}

function setBusyState(nextBusy) {
  isBusy = nextBusy;
  captureButton.disabled = nextBusy;
  renderVideoButton.disabled = nextBusy;
  loadMeshButton.disabled = nextBusy;
  refreshMeshesButton.disabled = nextBusy;
  viewerBadge.textContent = nextBusy ? "Working" : currentMeshPath ? "Ready" : "Idle";
}

function resizeRenderer() {
  const { clientWidth, clientHeight } = canvas;
  if (!clientWidth || !clientHeight) {
    return;
  }
  renderer.setSize(clientWidth, clientHeight, false);
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
}

function disposeCurrentObject() {
  if (!currentObject) {
    return;
  }
  currentObject.traverse((child) => {
    if (child.geometry) {
      child.geometry.dispose();
    }
    if (child.material) {
      if (Array.isArray(child.material)) {
        child.material.forEach((material) => material.dispose());
      } else {
        child.material.dispose();
      }
    }
  });
  scene.remove(currentObject);
  currentObject = null;
}

function fitCameraToObject(object) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const distance =
    maxDim > 0
      ? maxDim / Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5)) * 0.85
      : 2.5;

  const direction = new THREE.Vector3(1.1, 0.8, 1.1).normalize();
  camera.position.copy(center.clone().add(direction.multiplyScalar(distance)));
  camera.near = Math.max(distance / 200, 0.01);
  camera.far = distance * 200;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function applyPreviewNormalization(object, meshInfo) {
  const scale = meshInfo.scale > 0 ? 1 / meshInfo.scale : 1;
  object.scale.setScalar(scale);
  object.position.set(
    -meshInfo.centroid[0] * scale,
    -meshInfo.centroid[1] * scale,
    -meshInfo.centroid[2] * scale
  );
}

async function fetchJson(url, options = undefined) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

function renderMeshOptions() {
  const filter = meshSearchInput.value.trim().toLowerCase();
  const filtered = meshCatalog.filter((item) =>
    item.relative_path.toLowerCase().includes(filter)
  );

  meshList.innerHTML = "";
  for (const item of filtered) {
    const option = document.createElement("option");
    option.value = item.relative_path;
    option.textContent = item.relative_path;
    meshList.appendChild(option);
  }

  if (!meshList.value && filtered.length > 0) {
    const preferred = filtered.find((item) => item.relative_path === DEFAULT_MESH);
    meshList.value = preferred ? preferred.relative_path : filtered[0].relative_path;
  }

  if (!filtered.length) {
    setMeshStatus("No meshes match the current filter.");
  }
}

async function loadMeshCatalog() {
  setMeshStatus("Scanning mesh catalog...");
  meshCatalog = await fetchJson("/api/meshes");
  renderMeshOptions();
  const count = meshCatalog.length;
  setMeshStatus(count ? `Found ${count} meshes.` : "No meshes found.");
}

async function loadPreviewObject(meshPath) {
  const extension = meshPath.split(".").pop().toLowerCase();
  const assetUrl = `/api/mesh_asset?path=${encodeURIComponent(meshPath)}`;

  if (extension === "obj") {
    const loader = new OBJLoader();
    const object = await loader.loadAsync(assetUrl);
    object.traverse((child) => {
      if (child.isMesh) {
        child.material = new THREE.MeshStandardMaterial({
          color: 0xd7e3db,
          metalness: 0.05,
          roughness: 0.82,
        });
      }
    });
    return object;
  }

  if (extension === "ply") {
    const loader = new PLYLoader();
    const geometry = await loader.loadAsync(assetUrl);
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({
      color: 0xd7e3db,
      metalness: 0.05,
      roughness: 0.82,
    });
    return new THREE.Mesh(geometry, material);
  }

  throw new Error(`Unsupported preview extension: ${extension}`);
}

async function loadSelectedMesh() {
  const selectedPath = meshList.value;
  if (!selectedPath) {
    setMeshStatus("Select a mesh first.");
    return;
  }

  setBusyState(true);
  viewerOverlay.textContent = "Loading mesh...";
  viewerOverlay.style.display = "grid";
  setMeshStatus(`Loading ${selectedPath} ...`);

  try {
    const meshInfo = await fetchJson(
      `/api/mesh_info?path=${encodeURIComponent(selectedPath)}`
    );
    const object = await loadPreviewObject(selectedPath);

    disposeCurrentObject();
    applyPreviewNormalization(object, meshInfo);
    scene.add(object);
    currentObject = object;
    currentMeshPath = selectedPath;
    currentMeshInfo = meshInfo;

    fitCameraToObject(object);
    resizeRenderer();
    viewerTitle.textContent = selectedPath;
    viewerOverlay.style.display = "none";
    setMeshStatus(`Loaded ${selectedPath}`);
  } catch (error) {
    viewerOverlay.textContent = "Load failed.";
    viewerOverlay.style.display = "grid";
    setMeshStatus(`Load failed: ${error.message}`);
  } finally {
    setBusyState(false);
  }
}

function getViewerCameraPayload() {
  return {
    position: [camera.position.x, camera.position.y, camera.position.z],
    target: [controls.target.x, controls.target.y, controls.target.z],
  };
}

function syncPreviewFov() {
  camera.fov = Number(fovInput.value);
  camera.updateProjectionMatrix();
}

async function captureImage() {
  if (!currentMeshPath) {
    setRenderStatus("Load a mesh before capturing.");
    return;
  }

  setBusyState(true);
  setRenderStatus("Capturing still image...");
  try {
    const payload = {
      mesh_path: currentMeshPath,
      camera: getViewerCameraPayload(),
      image_size: Number(imageSizeInput.value),
      fov: Number(fovInput.value),
    };
    const record = await fetchJson("/api/capture_image", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    currentHistory.push(record);
    renderHistory(currentHistory);
    setRenderStatus("Still image captured.");
  } catch (error) {
    setRenderStatus(`Capture failed: ${error.message}`);
  } finally {
    setBusyState(false);
  }
}

async function renderVideo() {
  if (!currentMeshPath) {
    setRenderStatus("Load a mesh before rendering video.");
    return;
  }

  setBusyState(true);
  setRenderStatus("Rendering video with PyTorch3D...");
  try {
    const payload = {
      mesh_path: currentMeshPath,
      camera: getViewerCameraPayload(),
      trajectory_mode: trajectoryModeInput.value,
      duration_sec: Number(durationInput.value),
      fps: Number(fpsInput.value),
      image_size: Number(imageSizeInput.value),
      fov: Number(fovInput.value),
    };
    const record = await fetchJson("/api/render_video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    currentHistory.push(record);
    renderHistory(currentHistory);
    setRenderStatus("Video render completed.");
  } catch (error) {
    setRenderStatus(`Video render failed: ${error.message}`);
  } finally {
    setBusyState(false);
  }
}

function renderHistory(history) {
  historyContainer.innerHTML = "";
  if (!history.length) {
    historyContainer.innerHTML = '<div class="status">No captures yet.</div>';
    return;
  }

  history
    .slice()
    .reverse()
    .forEach((record) => {
      const card = document.createElement("article");
      card.className = "history-card";

      const title = document.createElement("h3");
      title.textContent =
        record.kind === "video" ? "Rendered Video" : "Captured Image";

      const meta = document.createElement("div");
      meta.className = "history-meta";
      meta.textContent = `${record.created_at}\n${record.mesh_path}`;

      const preview = document.createElement("div");
      preview.className = "history-preview";
      if (record.kind === "video" && record.video_url) {
        const video = document.createElement("video");
        video.src = record.video_url;
        video.controls = true;
        video.preload = "metadata";
        preview.appendChild(video);
      } else if (record.image_url) {
        const image = document.createElement("img");
        image.src = record.image_url;
        image.alt = record.record_id;
        preview.appendChild(image);
      }

      const links = document.createElement("div");
      links.className = "history-links";

      if (record.image_url) {
        const imageLink = document.createElement("a");
        imageLink.href = record.image_url;
        imageLink.textContent = "Open Image";
        imageLink.target = "_blank";
        links.appendChild(imageLink);
      }

      if (record.video_url) {
        const videoLink = document.createElement("a");
        videoLink.href = record.video_url;
        videoLink.textContent = "Open Video";
        videoLink.target = "_blank";
        links.appendChild(videoLink);
      }

      const metadataLink = document.createElement("a");
      metadataLink.href = record.metadata_url;
      metadataLink.textContent = "Metadata";
      metadataLink.target = "_blank";
      links.appendChild(metadataLink);

      card.appendChild(title);
      card.appendChild(meta);
      card.appendChild(preview);
      card.appendChild(links);
      historyContainer.appendChild(card);
    });
}

async function reloadHistory() {
  currentHistory = await fetchJson("/api/history");
  renderHistory(currentHistory);
}

async function clearHistory() {
  setRenderStatus("Clearing history...");
  try {
    const result = await fetchJson("/api/history/clear", { method: "POST" });
    currentHistory = [];
    renderHistory(currentHistory);
    setRenderStatus(`History cleared (${result.deleted_runs} runs).`);
  } catch (error) {
    setRenderStatus(`Failed to clear history: ${error.message}`);
  }
}

function resetView() {
  if (!currentObject) {
    return;
  }
  fitCameraToObject(currentObject);
}

function updateCameraReadout() {
  const position = camera.position;
  const target = controls.target;
  cameraReadout.textContent = [
    `position: [${position.x.toFixed(3)}, ${position.y.toFixed(3)}, ${position.z.toFixed(3)}]`,
    `target:   [${target.x.toFixed(3)}, ${target.y.toFixed(3)}, ${target.z.toFixed(3)}]`,
    `fov:      ${camera.fov.toFixed(1)}`,
  ].join("\n");
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  updateCameraReadout();
  renderer.render(scene, camera);
}

meshSearchInput.addEventListener("input", renderMeshOptions);
refreshMeshesButton.addEventListener("click", () => {
  loadMeshCatalog().catch((error) => setMeshStatus(`Mesh scan failed: ${error.message}`));
});
loadMeshButton.addEventListener("click", () => {
  loadSelectedMesh().catch((error) => setMeshStatus(`Load failed: ${error.message}`));
});
meshList.addEventListener("dblclick", () => {
  loadSelectedMesh().catch((error) => setMeshStatus(`Load failed: ${error.message}`));
});
resetViewButton.addEventListener("click", resetView);
captureButton.addEventListener("click", captureImage);
renderVideoButton.addEventListener("click", renderVideo);
clearHistoryButton.addEventListener("click", clearHistory);
fovInput.addEventListener("input", syncPreviewFov);
window.addEventListener("resize", resizeRenderer);

async function bootstrap() {
  resizeRenderer();
  syncPreviewFov();
  await loadMeshCatalog();
  await reloadHistory();
  if (meshCatalog.length > 0) {
    await loadSelectedMesh();
  }
}

viewerBadge.textContent = "Idle";
animate();
bootstrap().catch((error) => {
  setMeshStatus(`Bootstrap failed: ${error.message}`);
  viewerOverlay.textContent = "Initialization failed.";
  viewerOverlay.style.display = "grid";
});

