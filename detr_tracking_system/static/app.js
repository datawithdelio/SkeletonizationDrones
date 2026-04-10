const form = document.getElementById("upload-form");
const videoFileInput = document.getElementById("video-file");
const previewVideo = document.getElementById("preview-video");
const streamImage = document.getElementById("stream-image");
const streamPlaceholder = document.getElementById("stream-placeholder");
const formMessage = document.getElementById("form-message");
const streamState = document.getElementById("stream-state");
const metaSession = document.getElementById("meta-session");
const metaLabels = document.getElementById("meta-labels");
const metaMode = document.getElementById("meta-mode");

const statDevice = document.getElementById("stat-device");
const statFrames = document.getElementById("stat-frames");
const statTracks = document.getElementById("stat-tracks");
const statFps = document.getElementById("stat-fps");

let statusTimer = null;
let currentSessionId = null;
let currentPreviewUrl = null;

function setMessage(text, kind = "info") {
  formMessage.textContent = text;
  formMessage.dataset.kind = kind;
}

function setStreamState(label, tone) {
  streamState.textContent = label;
  streamState.className = `badge ${tone}`;
  document.body.dataset.stream = tone;
}

function updateStats(status) {
  statDevice.textContent = status.device ?? "-";
  statFrames.textContent = `${status.processed_frames ?? 0}${status.total_frames ? ` / ${status.total_frames}` : ""}`;
  statTracks.textContent = `${status.active_tracks ?? 0} active / ${status.unique_tracks_seen ?? 0} total`;
  statFps.textContent = Number(status.processing_fps ?? 0).toFixed(2);
  if (status.target_labels?.length) {
    metaLabels.textContent = status.target_labels.join(", ");
  }
  if (typeof status.drone_mode === "boolean") {
    metaMode.textContent = status.drone_mode ? "Tiled Sweep" : "Direct Detect";
  }
}

function stopStatusPolling() {
  if (statusTimer) {
    window.clearInterval(statusTimer);
    statusTimer = null;
  }
}

function resetViewer() {
  streamImage.classList.remove("is-visible");
  streamImage.removeAttribute("src");
  previewVideo.classList.remove("is-dimmed");
}

function attachPreview(file) {
  if (!(file instanceof File) || file.size === 0) {
    return;
  }

  if (currentPreviewUrl) {
    URL.revokeObjectURL(currentPreviewUrl);
  }

  currentPreviewUrl = URL.createObjectURL(file);
  previewVideo.src = currentPreviewUrl;
  previewVideo.classList.add("is-visible");
  previewVideo.classList.remove("is-dimmed");
  previewVideo.play().catch(() => {});
  streamPlaceholder.classList.add("is-hidden");
}

async function pollStatus(statusUrl) {
  try {
    const response = await fetch(statusUrl, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Status request failed with ${response.status}`);
    }

    const status = await response.json();
    updateStats(status);

    if (status.error) {
      setMessage(status.error, "error");
      setStreamState("Error", "error");
      previewVideo.classList.remove("is-dimmed");
      stopStatusPolling();
      return;
    }

    if (status.completed) {
      setMessage("Streaming completed. Reload the page or upload another video to start a new session.", "success");
      setStreamState("Complete", "success");
      stopStatusPolling();
      return;
    }

    if ((status.processed_frames ?? 0) > 0) {
      setStreamState("Streaming", "live");
    } else {
      setStreamState("Loading", "loading");
    }
  } catch (error) {
    setMessage(error.message, "error");
    setStreamState("Error", "error");
    previewVideo.classList.remove("is-dimmed");
    stopStatusPolling();
  }
}

videoFileInput.addEventListener("change", () => {
  const file = videoFileInput.files?.[0];
  if (!file) {
    return;
  }
  attachPreview(file);
  metaSession.textContent = "Pending";
  setStreamState("Idle", "idle");
  setMessage("Video selected. Start live tracking when you are ready.", "info");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  stopStatusPolling();

  const formData = new FormData(form);
  const file = formData.get("file");
  if (!(file instanceof File) || file.size === 0) {
    setMessage("Choose a video file before starting the tracker.", "error");
    return;
  }

  attachPreview(file);
  resetViewer();
  setMessage("Uploading video and warming the YOLO + DeepSORT pipeline. The first run may take a bit longer while model weights download.", "info");
  setStreamState("Loading", "loading");

  try {
    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || payload.note || "Upload failed.");
    }

    currentSessionId = payload.session_id;
    metaSession.textContent = currentSessionId.slice(0, 8).toUpperCase();
    metaLabels.textContent = payload.config.target_labels.join(", ");
    metaMode.textContent = payload.config.drone_mode ? "Tiled Sweep" : "Direct Detect";
    streamImage.src = `${payload.stream_url}?ts=${Date.now()}`;
    setMessage(
      `Session ${currentSessionId.slice(0, 8)} started. Tracking labels: ${payload.config.target_labels.join(", ")}. Drone mode: ${payload.config.drone_mode ? "on" : "off"}.`,
      "success",
    );

    statusTimer = window.setInterval(() => {
      pollStatus(payload.status_url);
    }, 1000);
    pollStatus(payload.status_url);
  } catch (error) {
    setMessage(error.message, "error");
    setStreamState("Error", "error");
    previewVideo.classList.remove("is-dimmed");
  }
});

streamImage.addEventListener("load", () => {
  streamImage.classList.add("is-visible");
  previewVideo.classList.add("is-dimmed");
  streamPlaceholder.classList.add("is-hidden");
});
