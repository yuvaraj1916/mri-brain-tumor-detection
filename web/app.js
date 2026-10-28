const form = document.querySelector("#analyze-form");
const fileInput = document.querySelector("#image-file");
const uploadZone = document.querySelector("#upload-zone");
const analyzeButton = document.querySelector("#analyze-button");
const resetButton = document.querySelector("#reset-button");
const selectedFile = document.querySelector("#selected-file");
const errorMessage = document.querySelector("#error-message");
const loadingStatus = document.querySelector("#loading-status");
const resultsSection = document.querySelector("#results");
const maxBytes = 10 * 1024 * 1024;
const extensions = new Set(["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]);
const classNames = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  notumor: "No Tumor",
  pituitary: "Pituitary",
};
let selectedImage = null;
let resultImages = null;

function setError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = !message;
}

function validateFile(file) {
  if (!file) return "Choose an MRI image to continue.";
  const extension = file.name.split(".").pop()?.toLowerCase();
  if (!extension || !extensions.has(extension)) {
    return "Unsupported format. Choose PNG, JPEG, BMP, TIFF, or WebP.";
  }
  if (file.size <= 0) return "The selected file is empty.";
  if (file.size > maxBytes) return "Image is too large. The maximum file size is 10 MB.";
  return "";
}

function setSelectedFile(file) {
  const error = validateFile(file);
  if (error) {
    selectedImage = null;
    fileInput.value = "";
    selectedFile.hidden = true;
    analyzeButton.disabled = true;
    setError(error);
    return;
  }
  selectedImage = file;
  setError("");
  document.querySelector("#file-name").textContent = file.name;
  document.querySelector("#file-size").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  selectedFile.hidden = false;
  uploadZone.hidden = true;
  analyzeButton.disabled = false;
}

function setLoading(isLoading) {
  analyzeButton.disabled = isLoading || !selectedImage;
  analyzeButton.classList.toggle("is-loading", isLoading);
  analyzeButton.querySelector(".button-text").textContent = isLoading ? "Analyzing…" : "Analyze MRI";
  loadingStatus.hidden = !isLoading;
}

function addClassProbabilities(probabilities) {
  const list = document.querySelector("#probability-list");
  list.replaceChildren();
  for (const [key, label] of Object.entries(classNames)) {
    if (!Object.hasOwn(probabilities, key)) continue;
    const value = Number(probabilities[key]);
    const item = document.createElement("div");
    item.className = "probability-item";
    const top = document.createElement("div");
    top.className = "probability-item-top";
    const name = document.createElement("span");
    name.textContent = label;
    const percent = document.createElement("strong");
    percent.textContent = `${(value * 100).toFixed(1)}%`;
    top.append(name, percent);
    const track = document.createElement("div");
    track.className = "probability-track";
    track.setAttribute("role", "meter");
    track.setAttribute("aria-label", `${label} probability`);
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", "100");
    track.setAttribute("aria-valuenow", String((value * 100).toFixed(1)));
    const fill = document.createElement("div");
    fill.className = "probability-fill";
    fill.style.width = `${Math.max(0, Math.min(100, value * 100))}%`;
    track.append(fill);
    item.append(top, track);
    list.append(item);
  }
}

function showResults(payload) {
  if (!payload || !classNames[payload.predicted_class] || !payload.class_probabilities
      || !payload.images || !payload.images.original || !payload.images.mask
      || !payload.images.overlay || payload.threshold !== 0.5
      || !Array.isArray(payload.image_info?.segmentation_processed_size)) {
    throw new Error("The server returned an incomplete analysis result.");
  }
  const expectedLabels = Object.keys(classNames);
  const probabilityEntries = Object.entries(payload.class_probabilities);
  if (probabilityEntries.length !== expectedLabels.length
      || probabilityEntries.some(([label, value], index) =>
        label !== expectedLabels[index] || !Number.isFinite(value) || value < 0 || value > 1)
      || Math.abs(probabilityEntries.reduce((total, [, value]) => total + value, 0) - 1) > 1e-5
      || payload.predicted_class !== probabilityEntries.reduce(
        (best, entry) => entry[1] > best[1] ? entry : best
      )[0]) {
    throw new Error("The server returned invalid classification probabilities.");
  }
  document.querySelector("#predicted-class").textContent = classNames[payload.predicted_class];
  addClassProbabilities(payload.class_probabilities);
  document.querySelector("#original-image").src = payload.images.original;
  document.querySelector("#predicted-mask").src = payload.images.mask;
  document.querySelector("#overlay-image").src = payload.images.overlay;
  document.querySelector("#threshold-label").textContent = `${payload.threshold.toFixed(2)} (configured)`;
  document.querySelector("#dimensions-label").textContent = payload.image_info.segmentation_processed_size.join(" × ");
  document.querySelector("#result-device").textContent = payload.image_info.device;
  document.querySelector("#device-label").textContent = `${payload.image_info.device} inference`;
  resultImages = payload.images;
  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

uploadZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => setSelectedFile(fileInput.files?.[0]));
document.querySelector("#remove-file").addEventListener("click", () => resetAll());

for (const eventName of ["dragenter", "dragover"]) {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.remove("is-dragging");
  });
}
uploadZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  setSelectedFile(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const validationError = validateFile(selectedImage);
  if (validationError) {
    setError(validationError);
    return;
  }
  setError("");
  resultsSection.hidden = true;
  setLoading(true);
  const body = new FormData();
  body.append("image", selectedImage, selectedImage.name);
  try {
    const response = await fetch("/api/analyze", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Analysis could not be completed.");
    showResults(payload);
  } catch (error) {
    setError(error instanceof TypeError
      ? "Could not reach the local analysis server. Check that it is running and try again."
      : error.message || "Analysis could not be completed. Try another image.");
  } finally {
    setLoading(false);
  }
});

function resetAll() {
  selectedImage = null;
  resultImages = null;
  fileInput.value = "";
  selectedFile.hidden = true;
  uploadZone.hidden = false;
  analyzeButton.disabled = true;
  resultsSection.hidden = true;
  setError("");
  setLoading(false);
  uploadZone.focus();
}
resetButton.addEventListener("click", resetAll);

document.querySelectorAll("[data-download]").forEach((button) => {
  button.addEventListener("click", () => {
    const key = button.dataset.download;
    if (!resultImages?.[key]) return;
    const anchor = document.createElement("a");
    anchor.href = resultImages[key];
    anchor.download = `mri-${key}.png`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
  });
});
