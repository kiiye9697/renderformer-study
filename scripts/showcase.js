document.querySelectorAll('[data-compare]').forEach(frame => {
  const control = frame.querySelector('input');
  const update = () => {
    frame.style.setProperty('--split', `${control.value}%`);
    control.setAttribute('aria-valuetext', `参考图占 ${control.value}%`);
  };
  control.addEventListener('input', update);
  update();
});
