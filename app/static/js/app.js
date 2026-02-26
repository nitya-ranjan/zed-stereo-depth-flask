/* ZED Stereo Depth Viewer – client-side logic */

(function () {
  "use strict";

  // ── DOM refs ────────────────────────────────────────────────
  const statusDot = document.getElementById("status-indicator");
  const statusText = document.getElementById("status-text");
  const colorFeed = document.getElementById("color-feed");
  const colorOverlay = document.getElementById("color-overlay");
  const cursorVal = document.getElementById("cursor-depth-value");

  const statMin = document.getElementById("stat-min");
  const statMax = document.getElementById("stat-max");
  const statMean = document.getElementById("stat-mean");
  const statMedian = document.getElementById("stat-median");
  const statStd = document.getElementById("stat-std");
  const statValid = document.getElementById("stat-valid");

  const obstacleGrid = document.getElementById("obstacle-grid");
  const thresholdInput = document.getElementById("obstacle-threshold");
  const thresholdDisplay = document.getElementById("threshold-display");

  const histCanvas = document.getElementById("histogram-canvas");
  const histCtx = histCanvas.getContext("2d");

  const detectionCount = document.getElementById("detection-count");
  const detectionsList = document.getElementById("detections-list");

  // Colour palette matching Python _PALETTE
  var PALETTE = [
    "#00ff80", "#ff8000", "#8000ff", "#00c8ff", "#ff0080",
    "#80ff00", "#ffff00", "#0080ff", "#ff00ff", "#00ffff",
  ];

  // ── Status polling ──────────────────────────────────────────
  function pollStatus() {
    fetch("/api/status")
      .then((r) => r.json())
      .then((d) => {
        statusDot.className = d.running ? "status connected" : "status disconnected";
        statusText.textContent = d.running ? "Camera active" : "Camera offline";
      })
      .catch(() => {
        statusDot.className = "status disconnected";
        statusText.textContent = "Server unreachable";
      });
  }

  // ── Depth stats ─────────────────────────────────────────────
  function pollStats() {
    fetch("/api/depth/stats")
      .then((r) => r.json())
      .then((d) => {
        if (d.error) return;
        statMin.textContent = d.min != null ? d.min + " m" : "-";
        statMax.textContent = d.max != null ? d.max + " m" : "-";
        statMean.textContent = d.mean != null ? d.mean + " m" : "-";
        statMedian.textContent = d.median != null ? d.median + " m" : "-";
        statStd.textContent = d.std != null ? d.std + " m" : "-";
        statValid.textContent =
          d.valid_ratio != null ? (d.valid_ratio * 100).toFixed(1) + "%" : "-";
      })
      .catch(() => {});
  }

  // ── Click-for-depth on colour feed ──────────────────────────
  colorFeed.parentElement.style.cursor = "crosshair";
  colorFeed.addEventListener("click", function (e) {
    var rect = colorFeed.getBoundingClientRect();
    var scaleX = colorFeed.naturalWidth / rect.width;
    var scaleY = colorFeed.naturalHeight / rect.height;
    var px = Math.round((e.clientX - rect.left) * scaleX);
    var py = Math.round((e.clientY - rect.top) * scaleY);

    // Draw crosshair
    var ctx = colorOverlay.getContext("2d");
    colorOverlay.width = colorOverlay.clientWidth;
    colorOverlay.height = colorOverlay.clientHeight;
    ctx.clearRect(0, 0, colorOverlay.width, colorOverlay.height);

    var dx = e.clientX - rect.left;
    var dy = e.clientY - rect.top;
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(dx - 12, dy);
    ctx.lineTo(dx + 12, dy);
    ctx.moveTo(dx, dy - 12);
    ctx.lineTo(dx, dy + 12);
    ctx.stroke();

    fetch("/api/depth/at?x=" + px + "&y=" + py)
      .then((r) => r.json())
      .then((d) => {
        cursorVal.textContent = d.depth_m != null ? d.depth_m.toFixed(3) : "-";
      })
      .catch(() => {
        cursorVal.textContent = "-";
      });
  });

  // ── Obstacle grid ───────────────────────────────────────────
  thresholdInput.addEventListener("input", function () {
    thresholdDisplay.textContent = thresholdInput.value;
  });

  function pollObstacles() {
    var threshold = parseFloat(thresholdInput.value);
    fetch("/api/depth/obstacles?threshold=" + threshold + "&rows=4&cols=6")
      .then((r) => r.json())
      .then((d) => {
        if (d.error) return;
        obstacleGrid.style.gridTemplateColumns = "repeat(6, 1fr)";
        obstacleGrid.innerHTML = "";
        d.grid.forEach(function (cell) {
          var el = document.createElement("div");
          el.className = "obstacle-cell " + (cell.is_obstacle ? "obstacle" : "clear");
          el.textContent = cell.mean_depth < 100 ? cell.mean_depth.toFixed(1) : "-";
          obstacleGrid.appendChild(el);
        });
      })
      .catch(() => {});
  }

  // ── Histogram ───────────────────────────────────────────────
  function pollHistogram() {
    fetch("/api/depth/histogram?bins=40")
      .then((r) => r.json())
      .then((d) => {
        if (d.error || !d.counts || d.counts.length === 0) return;
        drawHistogram(d.edges, d.counts);
      })
      .catch(() => {});
  }

  function drawHistogram(edges, counts) {
    var W = histCanvas.width;
    var H = histCanvas.height;
    histCtx.clearRect(0, 0, W, H);

    var max = Math.max.apply(null, counts) || 1;
    var barW = W / counts.length;

    for (var i = 0; i < counts.length; i++) {
      var barH = (counts[i] / max) * (H - 20);
      var x = i * barW;
      var y = H - 20 - barH;

      // Gradient from purple to red based on bin position (near=red, far=purple)
      var t = i / counts.length;
      var r = Math.round(108 + t * 123);
      var g = Math.round(92 - t * 50);
      var b = Math.round(231 - t * 170);
      histCtx.fillStyle = "rgb(" + r + "," + g + "," + b + ")";
      histCtx.fillRect(x + 1, y, barW - 2, barH);
    }

    // Axis labels
    histCtx.fillStyle = "#8b8fa3";
    histCtx.font = "10px sans-serif";
    histCtx.textAlign = "left";
    histCtx.fillText(edges[0].toFixed(1) + "m", 2, H - 4);
    histCtx.textAlign = "right";
    histCtx.fillText(edges[edges.length - 1].toFixed(1) + "m", W - 2, H - 4);
  }

  // ── Detections polling ──────────────────────────────────────
  function pollDetections() {
    fetch("/api/detections")
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.error) return;
        detectionCount.textContent = d.count + " object" + (d.count !== 1 ? "s" : "");
        detectionsList.innerHTML = "";
        d.detections.forEach(function (det) {
          var item = document.createElement("div");
          item.className = "det-item";

          var dot = document.createElement("span");
          dot.className = "det-color";
          dot.style.background = PALETTE[det.class_id % PALETTE.length];

          var name = document.createElement("span");
          name.className = "det-name";
          name.textContent = det.class_name;

          var conf = document.createElement("span");
          conf.className = "det-conf";
          conf.textContent = (det.confidence * 100).toFixed(0) + "%";

          var dist = document.createElement("span");
          dist.className = "det-dist";
          dist.textContent = det.distance_m != null ? det.distance_m.toFixed(1) + " m" : "- m";

          item.appendChild(dot);
          item.appendChild(name);
          item.appendChild(conf);
          item.appendChild(dist);
          detectionsList.appendChild(item);
        });
      })
      .catch(function () {});
  }

  // ── Polling intervals ───────────────────────────────────────
  pollStatus();
  setInterval(pollStatus, 3000);
  setInterval(pollStats, 1000);
  setInterval(pollObstacles, 1000);
  setInterval(pollHistogram, 1500);
  setInterval(pollDetections, 800);
  pollDetections();
})();
