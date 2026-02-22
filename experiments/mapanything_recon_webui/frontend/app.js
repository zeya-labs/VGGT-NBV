import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";
import { PLYLoader } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/PLYLoader.js";

const meshRootSelect = document.getElementById("mesh-root");
const refreshMeshesButton = document.getElementById("refresh-meshes");
const meshFilterInput = document.getElementById("mesh-filter");
const meshPathSelect = document.getElementById("mesh-path");
const meshCountEl = document.getElementById("mesh-count");

const numViewsInput = document.getElementById("num-views");
const seedInput = document.getElementById("seed");
const samplingModeSelect = document.getElementById("sampling-mode");
const cameraRadiusInput = document.getElementById("camera-radius");
const cameraVarInput = document.getElementById("camera-var");
const radiusModeSelect = document.getElementById("radius-mode");

const imageSizeInput = document.getElementById("image-size");
const fovInput = document.getElementById("fov");
const confThresholdInput = document.getElementById("conf-threshold");
const maxPointsInput = document.getElementById("max-points");
const showDepthInput = document.getElementById("show-depth");
const useDepthReconInput = document.getElementById("use-depth-recon");

const prepareButton = document.getElementById("prepare-btn");
const reconstructButton = document.getElementById("reconstruct-btn");
const clearHistoryButton = document.getElementById("clear-history");

const statusEl = document.getElementById("status");
const timingsEl = document.getElementById("timings");
const runIdBadgeEl = document.getElementById("run-id");
const pointStatsEl = document.getElementById("point-stats");

const rgbGallery = document.getElementById("rgb-gallery");
const depthGallery = document.getElementById("depth-gallery");
const historyEl = document.getElementById("history");

const pointCanvas = document.getElementById("pointcloud-canvas");

let currentRunId = null;
let allMeshPaths = [];
let meshListTruncated = false;
let currentPointCloud = null;

const pointScene = new THREE.Scene();
pointScene.background = new THREE.Color(0x061b1d);

const pointCamera = new THREE.PerspectiveCamera(60, 1, 0.001, 2000);
pointCamera.position.set(1.4, 1.1, 1.6);

const pointRenderer = new THREE.WebGLRenderer({
  canvas: pointCanvas,
  antialias: true,
  alpha: false,
});
pointRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const pointControls = new OrbitControls(pointCamera, pointRenderer.domElement);
pointControls.enableDamping = true;
pointControls.dampingFactor = 0.08;

pointScene.add(new THREE.AmbientLight(0xffffff, 0.9));

const plyLoader = new PLYLoader();

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "var(--danger)" : "var(--text)";
}

function setTimings(timings) {
  if (!timings || Object.keys(timings).length === 0) {
    timingsEl.textContent = "";
    return;
  }
  const lines = Object.entries(timings)
    .map(([k, v]) => `${k}: ${Number(v).toFixed(4)}s`)
    .join("\n");
  timingsEl.textContent = lines;
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload;
  try {
    payload = text ? JSON.parse(text) : {};
  } catch {
    payload = text;
  }

  if (!response.ok) {
    if (payload && typeof payload === "object" && "detail" in payload) {
      throw new Error(String(payload.detail));
    }
    throw new Error(typeof payload === "string" ? payload : `HTTP ${response.status}`);
  }
  return payload;
}

function createGalleryCard(url, title) {
  const item = document.createElement("div");
  item.className = "gallery-item";

  const img = document.createElement("img");
  img.src = url;
  img.loading = "lazy";
  img.alt = title;

  const caption = document.createElement("span");
  caption.textContent = title;

  item.appendChild(img);
  item.appendChild(caption);
  return item;
}

function renderGallery(container, urls, prefix) {
  container.innerHTML = "";
  if (!urls || urls.length === 0) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "No images";
    container.appendChild(empty);
    return;
  }

  urls.forEach((url, idx) => {
    container.appendChild(createGalleryCard(url, `${prefix} ${idx + 1}`));
  });
}

function resizePointViewer() {
  const width = pointCanvas.clientWidth;
  const height = pointCanvas.clientHeight;
  if (width <= 0 || height <= 0) {
    return;
  }

  pointRenderer.setSize(width, height, false);
  pointCamera.aspect = width / height;
  pointCamera.updateProjectionMatrix();
}

window.addEventListener("resize", resizePointViewer);

function animate() {
  requestAnimationFrame(animate);
  pointControls.update();
  pointRenderer.render(pointScene, pointCamera);
}

function fitCameraToObject(object3d) {
  const box = new THREE.Box3().setFromObject(object3d);
  if (box.isEmpty()) {
    return;
  }

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const distance = maxDim > 0 ? maxDim * 2.2 : 2.0;

  pointCamera.position.copy(center.clone().add(new THREE.Vector3(distance, distance * 0.8, distance)));
  pointCamera.near = Math.max(distance / 1000, 0.001);
  pointCamera.far = Math.max(distance * 50, 100);
  pointCamera.updateProjectionMatrix();

  pointControls.target.copy(center);
  pointControls.update();
}

function clearPointCloud() {
  if (!currentPointCloud) {
    return;
  }

  pointScene.remove(currentPointCloud);
  if (currentPointCloud.geometry) {
    currentPointCloud.geometry.dispose();
  }
  if (currentPointCloud.material) {
    currentPointCloud.material.dispose();
  }
  currentPointCloud = null;
}

async function loadPointCloud(plyUrl) {
  return new Promise((resolve, reject) => {
    plyLoader.load(
      plyUrl,
      (geometry) => {
        clearPointCloud();

        geometry.computeBoundingSphere();
        const material = new THREE.PointsMaterial({
          size: 0.006,
          sizeAttenuation: true,
          vertexColors: !!geometry.getAttribute("color"),
          color: geometry.getAttribute("color") ? 0xffffff : 0x9be7d6,
        });

        currentPointCloud = new THREE.Points(geometry, material);
        pointScene.add(currentPointCloud);
        fitCameraToObject(currentPointCloud);
        resolve();
      },
      undefined,
      (error) => reject(error)
    );
  });
}

function withCacheBust(url) {
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}v=${Date.now()}`;
}

function renderMeshOptions(paths) {
  meshPathSelect.innerHTML = "";
  if (!paths || paths.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No mesh files found";
    meshPathSelect.appendChild(option);
    return;
  }

  paths.forEach((path) => {
    const option = document.createElement("option");
    option.value = path;
    option.textContent = path;
    meshPathSelect.appendChild(option);
  });
}

function setMeshPlaceholder(text) {
  meshPathSelect.innerHTML = "";
  const option = document.createElement("option");
  option.value = "";
  option.textContent = text;
  meshPathSelect.appendChild(option);
}

function applyMeshFilter() {
  const keyword = meshFilterInput.value.trim().toLowerCase();
  const filtered = !keyword
    ? allMeshPaths
    : allMeshPaths.filter((path) => path.toLowerCase().includes(keyword));

  renderMeshOptions(filtered);
  const trunc = meshListTruncated ? " (truncated)" : "";
  meshCountEl.textContent = `${filtered.length} / ${allMeshPaths.length} mesh files${trunc}`;
}

async function loadMeshRoots() {
  const payload = await fetchJSON("/api/mesh_roots");
  const roots = payload.roots || [];

  meshRootSelect.innerHTML = "";
  roots.forEach((root, idx) => {
    const option = document.createElement("option");
    option.value = root.path;
    option.textContent = root.label || root.path;
    if (idx === 0) {
      option.selected = true;
    }
    meshRootSelect.appendChild(option);
  });
}

async function loadMeshList() {
  const root = meshRootSelect.value;
  if (!root) {
    allMeshPaths = [];
    setMeshPlaceholder("No mesh root selected");
    return;
  }

  setStatus("Listing meshes...");
  setMeshPlaceholder("Loading mesh list...");
  const url = `/api/mesh_list?root=${encodeURIComponent(root)}&limit=3000`;
  try {
    const payload = await fetchJSON(url);
    allMeshPaths = payload.meshes || [];
    meshListTruncated = !!payload.truncated;
    applyMeshFilter();
    setStatus(`Loaded ${allMeshPaths.length} mesh paths.`);
  } catch (error) {
    allMeshPaths = [];
    meshListTruncated = false;
    setMeshPlaceholder("Failed to load mesh list");
    throw error;
  }
}

function buildPreparePayload() {
  return {
    mesh_path: meshPathSelect.value,
    num_views: Number.parseInt(numViewsInput.value, 10),
    sampling: {
      view_sampling_mode: samplingModeSelect.value,
      seed: Number.parseInt(seedInput.value, 10),
      camera_radius: Number.parseFloat(cameraRadiusInput.value),
      camera_radius_variation: Number.parseFloat(cameraVarInput.value),
      camera_radius_mode: radiusModeSelect.value,
      up_axis: "Y",
      scene_index: 0,
      use_manual_camera: false,
    },
    render: {
      image_size: Number.parseInt(imageSizeInput.value, 10),
      fov: Number.parseFloat(fovInput.value),
      normalize_method: "unit_sphere",
      num_samples: 32768,
    },
    show_depth: showDepthInput.checked,
  };
}

async function handlePrepare() {
  if (!meshPathSelect.value) {
    setStatus("Please select a mesh path.", true);
    return;
  }
  const imageSize = Number.parseInt(imageSizeInput.value, 10);
  if (!Number.isFinite(imageSize) || imageSize % 14 !== 0) {
    setStatus("Image Size must be divisible by 14 (e.g. 518, 504, 560).", true);
    return;
  }

  prepareButton.disabled = true;
  reconstructButton.disabled = true;
  setStatus("Preparing sampled inputs with renderer...");
  setTimings({});

  try {
    const payload = buildPreparePayload();
    const result = await fetchJSON("/api/prepare_inputs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    currentRunId = result.run_id;
    runIdBadgeEl.textContent = `run: ${result.run_id}`;
    renderGallery(rgbGallery, result.rgb_urls, "RGB");
    renderGallery(depthGallery, result.depth_urls, "Depth");
    pointStatsEl.textContent = "points: -";
    setTimings(result.timings);
    reconstructButton.disabled = false;

    setStatus(`Prepared ${result.num_views} views on ${result.device}. Ready to reconstruct.`);
    await loadHistory();
  } catch (error) {
    setStatus(`Prepare failed: ${error.message}`, true);
  } finally {
    prepareButton.disabled = false;
  }
}

async function handleReconstruct() {
  if (!currentRunId) {
    setStatus("No prepared run. Click 'Prepare Inputs' first.", true);
    return;
  }

  reconstructButton.disabled = true;
  setStatus("Running MapAnything reconstruction. Please wait...");

  try {
    const useDepthInput = !!useDepthReconInput.checked;
    const payload = {
      run_id: currentRunId,
      conf_threshold: Number.parseFloat(confThresholdInput.value),
      max_points: Number.parseInt(maxPointsInput.value, 10),
      use_depth_input: useDepthInput,
    };

    const result = await fetchJSON("/api/reconstruct", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    await loadPointCloud(withCacheBust(result.ply_url));
    pointStatsEl.textContent = `points: ${result.num_points} (raw ${result.num_points_before_sampling})`;
    setTimings(result.timings);
    const depthMode = result.used_depth_input ? "enabled" : "disabled";
    setStatus(`Reconstruction done. Point cloud loaded. depth=${depthMode}.`);
    await loadHistory();
  } catch (error) {
    setStatus(`Reconstruct failed: ${error.message}`, true);
  } finally {
    reconstructButton.disabled = false;
  }
}

function applyHistoryRun(record) {
  currentRunId = record.run_id;
  runIdBadgeEl.textContent = `run: ${record.run_id}`;
  renderGallery(rgbGallery, record.rgb_urls || [], "RGB");
  renderGallery(depthGallery, record.depth_urls || [], "Depth");

  const reconstruct = record.reconstruct;
  if (reconstruct && reconstruct.ply_url) {
    pointStatsEl.textContent = `points: ${reconstruct.num_points}`;
    loadPointCloud(withCacheBust(reconstruct.ply_url))
      .then(() => setStatus(`Loaded point cloud from run ${record.run_id}`))
      .catch((err) => setStatus(`Failed to load point cloud: ${err.message}`, true));
  }

  reconstructButton.disabled = false;
}

function renderHistory(records) {
  historyEl.innerHTML = "";

  if (!records || records.length === 0) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "No history yet.";
    historyEl.appendChild(empty);
    return;
  }

  records
    .slice()
    .reverse()
    .forEach((record) => {
      const item = document.createElement("div");
      item.className = "history-item";

      const title = document.createElement("h3");
      title.textContent = `${record.run_id}`;

      const meta = document.createElement("div");
      meta.className = "history-meta";
      const rec = record.reconstruct;
      const pointInfo = rec ? ` | points=${rec.num_points}` : " | not reconstructed";
      const depthInfo = rec && typeof rec.use_depth_input === "boolean"
        ? ` | depth=${rec.use_depth_input ? "on" : "off"}`
        : "";
      meta.textContent = `${record.model_name} | views=${record.num_views}${pointInfo}${depthInfo}`;

      const actions = document.createElement("div");
      actions.className = "history-actions";

      const useButton = document.createElement("button");
      useButton.textContent = "Use";
      useButton.addEventListener("click", () => applyHistoryRun(record));
      actions.appendChild(useButton);

      if (rec && rec.ply_url) {
        const cloudButton = document.createElement("button");
        cloudButton.textContent = "Load Cloud";
        cloudButton.addEventListener("click", async () => {
          try {
            await loadPointCloud(withCacheBust(rec.ply_url));
            pointStatsEl.textContent = `points: ${rec.num_points}`;
            setStatus(`Loaded point cloud from run ${record.run_id}`);
          } catch (error) {
            setStatus(`Load cloud failed: ${error.message}`, true);
          }
        });
        actions.appendChild(cloudButton);
      }

      item.appendChild(title);
      item.appendChild(meta);
      item.appendChild(actions);
      historyEl.appendChild(item);
    });
}

async function loadHistory() {
  try {
    const records = await fetchJSON("/api/history");
    renderHistory(records);
  } catch (error) {
    setStatus(`Failed to load history: ${error.message}`, true);
  }
}

async function clearHistory() {
  try {
    const result = await fetchJSON("/api/history/clear", {
      method: "POST",
    });
    clearPointCloud();
    pointStatsEl.textContent = "points: -";
    currentRunId = null;
    runIdBadgeEl.textContent = "run: -";
    renderGallery(rgbGallery, [], "RGB");
    renderGallery(depthGallery, [], "Depth");
    setStatus(`History cleared (${result.deleted_runs} runs).`);
    await loadHistory();
  } catch (error) {
    setStatus(`Clear history failed: ${error.message}`, true);
  }
}

meshRootSelect.addEventListener("change", () => {
  loadMeshList().catch((err) => setStatus(`Failed to list meshes: ${err.message}`, true));
});

refreshMeshesButton.addEventListener("click", () => {
  loadMeshList().catch((err) => setStatus(`Failed to list meshes: ${err.message}`, true));
});

meshFilterInput.addEventListener("input", applyMeshFilter);
prepareButton.addEventListener("click", handlePrepare);
reconstructButton.addEventListener("click", handleReconstruct);
clearHistoryButton.addEventListener("click", clearHistory);

async function bootstrap() {
  resizePointViewer();
  animate();
  renderGallery(rgbGallery, [], "RGB");
  renderGallery(depthGallery, [], "Depth");

  try {
    await loadMeshRoots();
    await loadMeshList();
    await loadHistory();
    setStatus("Ready.");
  } catch (error) {
    setStatus(`Initialization failed: ${error.message}`, true);
  }
}

bootstrap();
