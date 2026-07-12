// 生成 Hero 底部的光谱条（呼应色彩聚类项目的视觉语言）
(function buildHeroSpectrum() {
  const el = document.getElementById("heroSpectrum");
  if (!el) return;

  // 一组随机但克制的色相，模拟"图片被拆解成若干主色"的效果
  const palette = [
    "#6FE7DD", "#3A8F89", "#B26EFF", "#FF8A65", "#FFC43D",
    "#5A9BFF", "#6FE7DD", "#B26EFF", "#3A8F89", "#FF8A65",
  ];
  const n = 24;
  for (let i = 0; i < n; i++) {
    const span = document.createElement("span");
    const color = palette[i % palette.length];
    const grow = 0.4 + Math.random() * 1.6;
    span.style.background = color;
    span.style.flexGrow = grow.toFixed(2);
    el.appendChild(span);
  }
})();

// 滚动进入视口时的淡入动效
(function scrollReveal() {
  const targets = document.querySelectorAll(".work-card");
  if (!("IntersectionObserver" in window)) {
    targets.forEach((t) => t.classList.add("in-view"));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );
  targets.forEach((t) => io.observe(t));
})();
