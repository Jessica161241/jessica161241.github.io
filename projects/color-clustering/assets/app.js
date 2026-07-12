/* 主逻辑：
   1. 图片读取
   2. 颜色空间转换 (RGB <-> LAB)
   3. K-means 聚类（含 k-means++ 初始化）
   4. 聚类结果可视化（ECharts：柱状图 / 饼图）+ 光谱条 + 数据表
   5. 调用 OpenAI 兼容接口，让大模型判断聚类颜色是否和谐
*/

// ---------------------------------------------------------------
// 全局状态
// ---------------------------------------------------------------
const state = {
  k: 5,
  colorSpace: "rgb",     // 'rgb' | 'lab'
  chartType: "bar",      // 'bar' | 'pie'
  imageEl: null,         // 当前 <img>，已加载完成
  pixels: null,          // 用于聚类的全像素 [{r,g,b}]
  clusters: null,        // 聚类结果 [{r,g,b,hex,count,pct}]
  chart: null,           // echarts 实例
};

const MAX_ITER = 30;
const TOL = 0.5;

// ---------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function setStatus(msg, type = "") {
  const el = $("#status");
  el.textContent = msg;
  el.className = "status" + (type ? " " + type : "");
}

// ---------------------------------------------------------------
// 颜色空间转换： sRGB -> CIE Lab
// 参考 D65 白点标准转换公式
// ---------------------------------------------------------------
function srgbToLinear(c) {
  c /= 255;
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function rgbToXyz(r, g, b) {
  const R = srgbToLinear(r), G = srgbToLinear(g), B = srgbToLinear(b);
  const X = R * 0.4124564 + G * 0.3575761 + B * 0.1804375;
  const Y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750;
  const Z = R * 0.0193339 + G * 0.1191920 + B * 0.9503041;
  return [X, Y, Z];
}

const D65 = [0.95047, 1.0, 1.08883];

function xyzToLab(x, y, z) {
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : (7.787 * t) + 16 / 116);
  const fx = f(x / D65[0]), fy = f(y / D65[1]), fz = f(z / D65[2]);
  const L = (116 * fy) - 16;
  const a = 500 * (fx - fy);
  const b = 200 * (fy - fz);
  return [L, a, b];
}

function rgbToLab(r, g, b) {
  const [x, y, z] = rgbToXyz(r, g, b);
  return xyzToLab(x, y, z);
}

// ---------------------------------------------------------------
// K-means
// points: [[a,b,c], ...]  在选定的颜色空间中的坐标
// ---------------------------------------------------------------
function dist2(p, q) {
  const dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
  return dx * dx + dy * dy + dz * dz;
}

function kmeansPlusPlusInit(points, k, rng) {
  const n = points.length;
  const centers = [points[Math.floor(rng() * n)]];
  while (centers.length < k) {
    const d2 = new Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      let m = Infinity;
      for (const c of centers) {
        const d = dist2(points[i], c);
        if (d < m) m = d;
      }
      d2[i] = m;
      sum += m;
    }
    if (sum === 0) {
      centers.push(points[Math.floor(rng() * n)]);
      continue;
    }
    let r = rng() * sum;
    let idx = 0;
    for (; idx < n; idx++) {
      r -= d2[idx];
      if (r <= 0) break;
    }
    centers.push(points[Math.min(idx, n - 1)]);
  }
  return centers.map((c) => c.slice());
}

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function kmeans(points, k, maxIter = MAX_ITER, tol = TOL) {
  const n = points.length;
  const rng = mulberry32(42); // 固定种子，保证同一图片/同一K结果可复现
  let centers = kmeansPlusPlusInit(points, k, rng);
  let labels = new Array(n).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;

    // -- 分配步骤 --
    for (let i = 0; i < n; i++) {
      let best = 0, bestD = Infinity;
      const p = points[i];
      for (let c = 0; c < k; c++) {
        const d = dist2(p, centers[c]);
        if (d < bestD) { bestD = d; best = c; }
      }
      if (labels[i] !== best) changed = true;
      labels[i] = best;
    }

    // -- 更新步骤 --
    const sums = Array.from({ length: k }, () => [0, 0, 0]);
    const counts = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      const c = labels[i], p = points[i];
      sums[c][0] += p[0]; sums[c][1] += p[1]; sums[c][2] += p[2];
      counts[c]++;
    }

    let maxShift = 0;
    const newCenters = centers.map((old, c) => {
      if (counts[c] === 0) {
        // 空簇：重新在最远的点上放置新中心，避免簇消失
        let far = 0, farD = -1;
        for (let i = 0; i < n; i++) {
          let m = Infinity;
          for (let cc = 0; cc < k; cc++) m = Math.min(m, dist2(points[i], centers[cc]));
          if (m > farD) { farD = m; far = i; }
        }
        return points[far].slice();
      }
      const nc = [sums[c][0] / counts[c], sums[c][1] / counts[c], sums[c][2] / counts[c]];
      maxShift = Math.max(maxShift, Math.sqrt(dist2(nc, old)));
      return nc;
    });

    centers = newCenters;
    if (!changed || maxShift < tol) break;
  }

  return { centers, labels };
}

// ---------------------------------------------------------------
// 图片 -> 像素
// ---------------------------------------------------------------
function extractPixels(imgEl) {
   const w = imgEl.naturalWidth;
  const h = imgEl.naturalHeight;

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(imgEl, 0, 0, w, h);

  const { data } = ctx.getImageData(0, 0, w, h);
  const pixels = [];
  for (let i = 0; i < data.length; i += 4) {
    const alpha = data[i + 3];
    if (alpha < 128) continue; // 忽略透明像素
    pixels.push({ r: data[i], g: data[i + 1], b: data[i + 2] });
  }
  return pixels;
}

// ---------------------------------------------------------------
// 运行聚类 + 汇总结果
// ---------------------------------------------------------------
function runClustering() {
  if (!state.pixels || state.pixels.length === 0) {
    setStatus("请先选择或上传一张图片", "error");
    return;
  }
  const k = state.k;
  if (state.pixels.length < k) {
    setStatus("像素数量少于 K 值，请降低 K", "error");
    return;
  }

  if (typeof echarts === "undefined") {
    setStatus("图表库（ECharts）加载失败，请检查网络连接后刷新页面重试", "error");
    return;
  }

  setStatus("聚类计算中 …");

  try {
    const points = state.pixels.map((p) => {
      if (state.colorSpace === "lab") return rgbToLab(p.r, p.g, p.b);
      return [p.r, p.g, p.b];
    });

    const { labels } = kmeans(points, k);

    // 按簇统计像素数量与 RGB 均值（均值始终在 RGB 空间中计算，
    // 因为题目要求可视化"类簇中像素颜色均值"，这样无论聚类发生在
    // RGB 还是 Lab 空间，展示的都是可直接理解的真实颜色均值）
    const sums = Array.from({ length: k }, () => [0, 0, 0]);
    const counts = new Array(k).fill(0);
    for (let i = 0; i < state.pixels.length; i++) {
      const c = labels[i], p = state.pixels[i];
      sums[c][0] += p.r; sums[c][1] += p.g; sums[c][2] += p.b;
      counts[c]++;
    }

    const total = state.pixels.length;
    let clusters = counts.map((count, i) => {
      const r = count ? Math.round(sums[i][0] / count) : 0;
      const g = count ? Math.round(sums[i][1] / count) : 0;
      const b = count ? Math.round(sums[i][2] / count) : 0;
      return {
        r, g, b,
        hex: rgbToHex(r, g, b),
        count,
        pct: count / total,
      };
    });

    // 按占比从大到小排序，方便阅读
    clusters = clusters.filter((c) => c.count > 0).sort((a, b) => b.count - a.count);

    state.clusters = clusters;
    renderAll();
    setStatus(`完成：K=${k}，颜色空间=${state.colorSpace.toUpperCase()}，采样像素 ${total}`, "ok");
  } catch (err) {
    console.error(err);
    setStatus("聚类过程中出错：" + err.message, "error");
  }
}

function rgbToHex(r, g, b) {
  return "#" + [r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("").toUpperCase();
}

// ---------------------------------------------------------------
// 渲染：光谱条 + 表格 + 图表
// ---------------------------------------------------------------
function renderAll() {
  renderSpectrum();
  renderTable();
  renderChart();
}

function renderSpectrum() {
  const strip = $("#spectrum");
  strip.innerHTML = "";
  state.clusters.forEach((c) => {
    const seg = document.createElement("div");
    seg.className = "spectrum-seg";
    seg.style.flexGrow = c.pct;
    seg.style.background = c.hex;
    seg.innerHTML = `<div class="tip">${c.hex} · ${(c.pct * 100).toFixed(1)}% · ${c.count}px</div>`;
    strip.appendChild(seg);
  });
}

function renderTable() {
  const tbody = $("#clusterTable tbody");
  tbody.innerHTML = "";
  state.clusters.forEach((c, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">C${i + 1}</td>
      <td><span class="swatch" style="background:${c.hex}"></span><span class="mono">${c.hex}</span></td>
      <td class="mono">${c.r}, ${c.g}, ${c.b}</td>
      <td class="mono">${c.count.toLocaleString()}</td>
      <td class="mono">${(c.pct * 100).toFixed(1)}%</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderChart() {
  if (!state.chart) {
    state.chart = echarts.init($("#chart"), null, { renderer: "canvas" });
    window.addEventListener("resize", () => state.chart.resize());
  }

  const labels = state.clusters.map((c, i) => `C${i + 1} ${c.hex}`);
  const values = state.clusters.map((c) => c.count);
  const colors = state.clusters.map((c) => c.hex);

  let option;

  if (state.chartType === "bar") {
    option = {
      backgroundColor: "transparent",
      textStyle: { color: "#ECEAE4", fontFamily: "IBM Plex Mono, monospace" },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: "#21232C",
        borderColor: "#33353F",
        textStyle: { color: "#ECEAE4" },
        formatter: (params) => {
          const p = params[0];
          const c = state.clusters[p.dataIndex];
          return `${labels[p.dataIndex]}<br/>像素数：${c.count.toLocaleString()}<br/>占比：${(c.pct * 100).toFixed(1)}%`;
        },
      },
      grid: { left: 50, right: 20, top: 40, bottom: 60 },
      xAxis: {
        type: "category",
        data: labels,
        axisLine: { lineStyle: { color: "#33353F" } },
        axisLabel: { color: "#9A9CA8", rotate: 30, fontSize: 11 },
      },
      yAxis: {
        type: "value",
        name: "像素数量",
        nameTextStyle: { color: "#9A9CA8" },
        axisLine: { show: false },
        splitLine: { lineStyle: { color: "#242631" } },
        axisLabel: { color: "#9A9CA8" },
      },
      series: [
        {
          type: "bar",
          data: values.map((v, i) => ({
            value: v,
            itemStyle: { color: colors[i], 
              borderRadius: [3, 3, 0, 0], 
              borderColor: 'rgba(236, 234, 228, 0.5)', // 半透明白色描边
              borderWidth: 1.5},
          })),
          barMaxWidth: 46,
        },
      ],
    };
  } else {
    option = {
      backgroundColor: "transparent",
      textStyle: { color: "#ECEAE4", fontFamily: "IBM Plex Mono, monospace" },
      tooltip: {
        trigger: "item",
        backgroundColor: "#21232C",
        borderColor: "#33353F",
        textStyle: { color: "#ECEAE4" },
        formatter: (p) => {
          const c = state.clusters[p.dataIndex];
          return `${labels[p.dataIndex]}<br/>像素数：${c.count.toLocaleString()}<br/>占比：${(c.pct * 100).toFixed(1)}%`;
        },
      },
      legend: {
        bottom: 0,
        textStyle: { color: "#9A9CA8", fontSize: 11 },
        icon: "circle",
      },
      series: [
        {
          type: "pie",
          radius: ["36%", "68%"],
          center: ["50%", "46%"],
          avoidLabelOverlap: true,
          itemStyle: {
            borderColor: 'rgba(236, 234, 228, 0.4)',
            borderWidth: 1.5
          },
          label: {
            color: "#ECEAE4",
            fontSize: 11,
            formatter: "{b}\n{d}%",
          },
          labelLine: { lineStyle: { color: "#33353F" } },
          data: state.clusters.map((c, i) => ({
            value: c.count,
            name: labels[i],
            itemStyle: { color: c.hex },
          })),
        },
      ],
    };
  }

  state.chart.setOption(option, true);
}

// ---------------------------------------------------------------
// 图片加载
// ---------------------------------------------------------------
function loadImageFromSrc(src, done) {
  const img = new Image();
  img.onload = () => {
    state.imageEl = img;
    const wrap = $("#previewWrap");
    wrap.innerHTML = "";
    wrap.appendChild(img);
    state.pixels = extractPixels(img);
    setStatus(`已载入图片：${img.naturalWidth}×${img.naturalHeight}px`, "ok");
    if (done) done();
  };
  img.onerror = () => setStatus("图片加载失败", "error");
  img.src = src;
}

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("请选择图片文件", "error");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    loadImageFromSrc(e.target.result, () => runClustering());
    $$(".sample-thumb").forEach((el) => el.classList.remove("active"));
  };
  reader.readAsDataURL(file);
}

// ---------------------------------------------------------------
// AI 颜色和谐度判断
// ---------------------------------------------------------------
async function callHarmonyAI() {
  const baseUrl = $("#aiBaseUrl").value.trim().replace(/\/+$/, "");
  const apiKey = $("#aiApiKey").value.trim();
  const model = $("#aiModel").value.trim();

  if (!state.clusters || state.clusters.length === 0) {
    setAiStatus("请先完成聚类", "error");
    return;
  }
  if (!baseUrl || !apiKey) {
    setAiStatus("请填写 API Base URL 与 API Key", "error");
    return;
  }

  const resultBox = $("#aiResult");
  resultBox.textContent = "AI 分析中 …";
  resultBox.classList.remove("ready");
  setAiStatus("请求已发送 …");

  const colorList = state.clusters
    .map((c, i) => `C${i + 1}: ${c.hex} (RGB ${c.r},${c.g},${c.b}, 占比 ${(c.pct * 100).toFixed(1)}%)`)
    .join("\n");

  const prompt =
    `以下是从一张图片中通过 K-means 聚类得到的 ${state.clusters.length} 个主要颜色（按像素占比排序）：\n` +
    `${colorList}\n\n` +
    `请你作为配色设计师，判断这些颜色搭配在一起是否和谐，并说明理由（可以从色相关系、明度对比、饱和度等角度分析）。` +
    `最后给出一个"和谐度评分"（0-10分）。请用简洁的中文回答，控制在200字以内。`;

  try {
    const resp = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: "你是一名专业的色彩与视觉设计顾问。" },
          { role: "user", content: prompt },
        ],
        temperature: 0.6,
      }),
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${text.slice(0, 200)}`);
    }

    const data = await resp.json();
    const content =
      data?.choices?.[0]?.message?.content ??
      data?.choices?.[0]?.text ??
      JSON.stringify(data);

    let formatted = content.trim();
formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
resultBox.innerHTML = formatted;
    resultBox.classList.add("ready");
    setAiStatus("完成", "ok");
  } catch (err) {
    resultBox.textContent = "";
    resultBox.classList.remove("ready");
    setAiStatus("请求失败：" + err.message + "（请检查接口地址 / Key / 跨域设置）", "error");
  }
}

function setAiStatus(msg, type = "") {
  const el = $("#aiStatus");
  el.textContent = msg;
  el.className = "status" + (type ? " " + type : "");
}

// ---------------------------------------------------------------
// UI 绑定
// ---------------------------------------------------------------
function initSamples() {
  const grid = $("#sampleGrid");
  SAMPLE_IMAGES.forEach((s, idx) => {
    const el = document.createElement("div");
    el.className = "sample-thumb" + (idx === 0 ? " active" : "");
    el.style.backgroundImage = `url(${s.src})`;
    el.tabIndex = 0;
    el.title = s.label;
    el.innerHTML = `<span>${s.label}</span>`;
    el.addEventListener("click", () => {
      $$(".sample-thumb").forEach((n) => n.classList.remove("active"));
      el.classList.add("active");
      loadImageFromSrc(s.src, () => runClustering());
    });
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") el.click();
    });
    grid.appendChild(el);
  });
}

function initControls() {
  // K 值滑杆
  const kRange = $("#kRange");
  const kVal = $("#kVal");
  kRange.addEventListener("input", () => { kVal.textContent = kRange.value; });
  kRange.addEventListener("change", () => {
    state.k = parseInt(kRange.value, 10);
    runClustering();
  });

  // 颜色空间
  $$("#colorSpaceSeg button").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$("#colorSpaceSeg button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.colorSpace = btn.dataset.value;
      runClustering();
    });
  });

  // 图表类型
  $$("#chartTypeSeg button").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$("#chartTypeSeg button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.chartType = btn.dataset.value;
      if (state.clusters) renderChart();
    });
  });

  // 上传
  const dropzone = $("#dropzone");
  const fileInput = $("#fileInput");
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") fileInput.click();
  });
  fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  });

  // AI 面板
  $("#aiRunBtn").addEventListener("click", callHarmonyAI);
}

function init() {
  initSamples();
  initControls();
  // 默认载入第一张示例图
  loadImageFromSrc(SAMPLE_IMAGES[0].src, () => runClustering());
}

document.addEventListener("DOMContentLoaded", init);

// 兜底：任何未被 try/catch 捕获的运行时错误都在状态栏里显示出来，
// 避免界面卡在"计算中"却没有任何提示。
window.addEventListener("error", (e) => {
  setStatus("页面出错：" + e.message + "（请打开浏览器控制台 F12 查看详情）", "error");
});
