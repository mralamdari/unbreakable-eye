/**
 * Zone overlay display for the monitor/video wall page.
 * Read-only — fetches zones per camera and renders polygon overlays on each tile.
 */

(function () {
  function hexToRgba(hex, alpha) {
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
  }

  function drawZonesOnCanvas(canvas, zones) {
    var ctx = canvas.getContext('2d');
    var w = canvas.width;
    var h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    zones.forEach(function (zone, idx) {
      var poly = zone.polygon;
      if (!poly || poly.length < 3) return;
      var color = zone.color || '#4f8cff';

      // Fill polygon
      ctx.beginPath();
      ctx.moveTo(poly[0][0] * w, poly[0][1] * h);
      for (var i = 1; i < poly.length; i++) {
        ctx.lineTo(poly[i][0] * w, poly[i][1] * h);
      }
      ctx.closePath();
      ctx.fillStyle = hexToRgba(color, 0.15);
      ctx.fill();

      // Border
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Zone number badge at centroid
      var cx = poly.reduce(function (s, p) { return s + p[0]; }, 0) / poly.length * w;
      var cy = poly.reduce(function (s, p) { return s + p[1]; }, 0) / poly.length * h;
      var label = String(idx + 1);

      ctx.beginPath();
      ctx.arc(cx, cy, 10, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, cx, cy);
    });
  }

  function sizeCanvasToTile(canvas, tile) {
    var img = tile.querySelector('img');
    if (!img) return false;
    var w = img.clientWidth;
    var h = img.clientHeight;
    if (!w || !h) return false;
    canvas.width = w;
    canvas.height = h;
    return true;
  }

  function init() {
    var tiles = document.querySelectorAll('.tile[data-cam-id]');
    tiles.forEach(function (tile) {
      var camId = tile.dataset.camId;
      var canvas = tile.querySelector('.zone-overlay');
      if (!canvas) return;

      fetch('/api/zones/' + camId)
        .then(function (r) { return r.json(); })
        .then(function (data) {
          var zones = data.zones || [];
          if (zones.length === 0) return;

          if (sizeCanvasToTile(canvas, tile)) {
            drawZonesOnCanvas(canvas, zones);
          }

          // Re-render on resize
          window.addEventListener('resize', function () {
            if (sizeCanvasToTile(canvas, tile)) {
              drawZonesOnCanvas(canvas, zones);
            }
          });
        })
        .catch(function () {});
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
