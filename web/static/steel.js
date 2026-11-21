const modelSelector = document.querySelector(".model-selector");

if (modelSelector) {
  const closeButton = modelSelector.querySelector("[data-close-model]");
  const closeModelSelector = () => {
    modelSelector.open = false;
    modelSelector.querySelector("summary").focus();
  };
  closeButton.hidden = false;
  closeButton.addEventListener("click", closeModelSelector);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && modelSelector.open) {
      closeModelSelector();
    }
  });
  document.addEventListener("click", (event) => {
    if (modelSelector.open && !modelSelector.contains(event.target)) {
      modelSelector.open = false;
    }
  });
}

document.addEventListener("submit", (event) => {
  const button = event.submitter;
  if (button?.dataset.busy) {
    button.dataset.originalLabel = button.textContent;
    button.textContent = button.dataset.busy;
    button.disabled = true;
  }
});

window.addEventListener("pageshow", () => {
  document.querySelectorAll("[data-original-label]").forEach((button) => {
    button.textContent = button.dataset.originalLabel;
    button.disabled = false;
    delete button.dataset.originalLabel;
  });
});
