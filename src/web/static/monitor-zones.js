/**
 * Zone overlay display for the monitor/video wall page.
 * Read-only — fetches zones per camera and renders polygon overlays on each tile.
 * Zone labels are HTML elements (not canvas) so they scale crisply on zoom.
 */

(function () {
  function hexToRgba(hex, alpha) {
    var r = parseInt(hex.slice(1, 3), 16);
    var g = parseInt(hex.slice(3, 5), 16);
    var b = parseInt(hex.slice(5, 7), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
  }

  function drawPolygonsOnCanvas(canvas, zones) {
    var ctx = canvas.getContext('2d');
    var w = canvas.width;
    var h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    zones.forEach(function (zone) {
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
    });
  }

  var LABEL_VERTICAL_OFFSET = 6;

  function getZoneTopRight(zone, w, h) {
    var poly = zone.polygon;
    var maxX = -Infinity, minY = Infinity;
    poly.forEach(function (p) {
      if (p[0] * w > maxX) maxX = p[0] * w;
      if (p[1] * h < minY) minY = p[1] * h;
    });
    return { x: maxX, y: minY };
  }

  function createLabelElements(tile, zones) {
    // Remove old labels
    tile.querySelectorAll('.zone-label').forEach(function (el) { el.remove(); });

    zones.forEach(function (zone, idx) {
      var div = document.createElement('div');
      div.className = 'zone-label';
      div.textContent = zone.name || 'Zone ' + (idx + 1);
      div.style.backgroundColor = zone.color || '#4f8cff';
      tile.appendChild(div);
    });
  }

  function positionLabels(tile, zones, canvasW, canvasH) {
    var labels = tile.querySelectorAll('.zone-label');
    var canvas = tile.querySelector('.zone-overlay');
    var canvasLeft = parseInt(canvas.style.left) || 0;
    var canvasTop = parseInt(canvas.style.top) || 0;

    labels.forEach(function (label, idx) {
      var zone = zones[idx];
      if (!zone) return;
      var pos = getZoneTopRight(zone, canvasW, canvasH);
      label.style.left = (canvasLeft + pos.x - 4) + 'px';
      label.style.top = (canvasTop + pos.y + 4) + 'px';
    });
  }

  function sizeCanvasToTile(canvas, tile) {
    var img = tile.querySelector('img');
    if (!img) return false;
    var w = img.clientWidth;
    var h = img.clientHeight;
    if (!w || !h) return false;

    // Account for object-fit: contain letterboxing when zoomed
    var isZoomed = tile.classList.contains('zoom');
    if (isZoomed) {
      var imgW = img.naturalWidth || w;
      var imgH = img.naturalHeight || h;
      var imgRatio = imgW / imgH;
      var containerRatio = w / h;
      var visibleW, visibleH, offsetX, offsetY;

      if (imgRatio > containerRatio) {
        visibleW = w;
        visibleH = w / imgRatio;
        offsetX = 0;
        offsetY = (h - visibleH) / 2;
      } else {
        visibleH = h;
        visibleW = h * imgRatio;
        offsetX = (w - visibleW) / 2;
        offsetY = 0;
      }

      canvas.width = Math.round(visibleW);
      canvas.height = Math.round(visibleH);
      canvas.style.left = Math.round(offsetX) + 'px';
      canvas.style.top = Math.round(offsetY) + 'px';
    } else {
      canvas.width = w;
      canvas.height = h;
      canvas.style.left = '0px';
      canvas.style.top = '0px';
    }
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

          createLabelElements(tile, zones);

          function render() {
            if (sizeCanvasToTile(canvas, tile)) {
              drawPolygonsOnCanvas(canvas, zones);
              positionLabels(tile, zones, canvas.width, canvas.height);
            }
          }

          render();
          window.addEventListener('resize', render);

          // Re-render when zoom class changes
          var observer = new MutationObserver(function (mutations) {
            mutations.forEach(function (mutation) {
              if (mutation.attributeName === 'class') {
                render();
              }
            });
          });
          observer.observe(tile, { attributes: true, attributeFilter: ['class'] });

          // Also render when image loads (handles async image loading)
          var img = tile.querySelector('img');
          if (img) {
            img.addEventListener('load', render);
            if (img.complete) {
              render();
            }
          }
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
