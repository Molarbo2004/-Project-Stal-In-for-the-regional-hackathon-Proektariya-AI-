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

const workspaceSwitch = document.querySelector(".workspace-switch");
if (workspaceSwitch) {
  const mobile = window.matchMedia("(max-width: 779px)");
  const dashboard = document.getElementById("dashboard");
  const recognition = document.getElementById("recognition");
  function updateWorkspace() {
    const target = document.getElementById(location.hash.slice(1));
    const showDashboard = target && dashboard.contains(target);
    recognition.hidden = mobile.matches && Boolean(showDashboard);
    dashboard.hidden = mobile.matches && !showDashboard;
    for (const link of workspaceSwitch.querySelectorAll("a")) {
      if (link.hash === (showDashboard ? "#dashboard" : "#recognition")) {
        link.setAttribute("aria-current", "location");
      } else {
        link.removeAttribute("aria-current");
      }
    }
  }
  function navigateWorkspace() {
    updateWorkspace();
    if (mobile.matches && ["#recognition", "#dashboard"].includes(location.hash)) {
      window.scrollTo(0, 0);
    }
  }
  window.addEventListener("hashchange", navigateWorkspace);
  mobile.addEventListener("change", updateWorkspace);
  navigateWorkspace();
}


const reviewPosition = document.querySelector("#position");
if (reviewPosition) {
  const form = reviewPosition.form;
  reviewPosition.addEventListener("change", () => form.requestSubmit());
  form.querySelector('button[type="submit"]').hidden = true;
}
