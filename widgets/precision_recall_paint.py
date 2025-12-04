# # Interactive contrail mask click-paint with live metrics (single-cell version)
# # ---------------------------------------------------------------------------
# # If you're on JupyterLab, make sure you have ipympl + ipywidgets installed:
# #   pip install ipympl ipywidgets
# # This cell will try to enable an interactive backend automatically.
# #
# # Ground truth: pixels that are full black (0 or (0,0,0)) in the ORIGINAL image are POSITIVE.
# # Prediction: pixels you paint green are predicted positives.
# # Metrics update live: Sensitivity, Specificity, Precision, NPV, Accuracy.
# #
# # NOTE: If your source is a .XCF, export it to PNG/JPG first (e.g., florida_2020_03_05_0101.png)
# # and set IMAGE_PATH below.

# # --- backend setup (must run before importing pyplot) ---
# from IPython import get_ipython
# _ip = get_ipython()
# _backend_set = False
# if _ip is not None:
#     try:
#         import ipympl  # noqa: F401
#         _ip.run_line_magic("matplotlib", "widget")
#         _backend_set = True
#     except Exception:
#         pass
#     if not _backend_set:
#         try:
#             _ip.run_line_magic("matplotlib", "notebook")  # works in classic Jupyter Notebook
#             _backend_set = True
#         except Exception:
#             pass

# # --- imports ---
# from PIL import Image, UnidentifiedImageError
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Cursor
# from ipywidgets import HBox, VBox, Button, IntSlider, ToggleButton, HTML, Text, Output
# from IPython.display import display, clear_output
# import os

# # ==== User settings ====

# DEFAULT_BRUSH_RADIUS = 5
# BLACK_TOLERANCE = 127  # treat values <= this as black; 0 = exact black only
# # =======================

# out = Output()

# def load_image(path):
#     try:
        
#         img = Image.open(path)
#     except (FileNotFoundError, UnidentifiedImageError) as e:
#         raise RuntimeError(
#             f"Could not open image '{path}'. If your source is a .xcf, export it to PNG/JPG first.\n"
#             f"Original error: {e}"
#         )
#     if img.mode in ("RGBA", "RGB"):
#         arr = np.array(img.convert("RGB"))
#     elif img.mode in ("L", "I;16", "I", "F"):
#         arr = np.array(img.convert("L"))
#     else:
#         arr = np.array(img.convert("RGB"))
#     return arr


# def load_label(path):
#     try:
        
#         img = Image.open(path)
#     except (FileNotFoundError, UnidentifiedImageError) as e:
#         raise RuntimeError(
#             f"Could not open image '{path}'. If your source is a .xcf, export it to PNG/JPG first.\n"
#             f"Original error: {e}"
#         )
#     label = np.array(img.convert("RGBA"))
    

#     # Alpha channel is label
#     alpha = label[:, :, 3]

#     return alpha




# def compute_gt_mask(img_arr, black_tol=0):
#     if img_arr.ndim == 2:  # grayscale
#         gt_pos = img_arr >= black_tol
#     else:  # RGB
#         gt_pos = (img_arr[...,0] <= black_tol) & (img_arr[...,1] <= black_tol) & (img_arr[...,2] <= black_tol)
#     return gt_pos 

# def metrics_from_masks(pred_pos: np.ndarray, gt_pos: np.ndarray):
#     tp = int(np.sum(pred_pos & gt_pos))
#     fp = int(np.sum(pred_pos & ~gt_pos))
#     tn = int(np.sum(~pred_pos & ~gt_pos))
#     fn = int(np.sum(~pred_pos & gt_pos))
#     total = tp + fp + tn + fn
#     def safe_div(n, d): return float(n)/float(d) if d != 0 else np.nan
#     return {
#         "TP": tp, "FP": fp, "TN": tn, "FN": fn, "Total": total,
#         "Sensitivity": safe_div(tp, tp + fn),
#         "Specificity": safe_div(tn, tn + fp),
#         "Precision":   safe_div(tp, tp + fp),
#         "NPV":         safe_div(tn, tn + fn),
#         "Accuracy":    safe_div(tp + tn, total),
#     }

# def fmt_pct(x): return "—" if (x is None or np.isnan(x)) else f"{x*100:.2f}%"

# class Painter:
#     def __init__(self, image_path, label_path, brush_radius=DEFAULT_BRUSH_RADIUS, black_tol=BLACK_TOLERANCE):
#         self.image_path = image_path
#         self.img = load_image(image_path)
#         self.h, self.w = self.img.shape[:2]
#         self.gt = compute_gt_mask(load_label(label_path), black_tol=black_tol)  # ground-truth positives (black)

#         self.pred = np.zeros((self.h, self.w), dtype=bool)        # predicted positives (painted)

#         self.fig, self.ax = plt.subplots(figsize=(12, 12 * self.h / self.w) if self.w else (8, 8))
#         self.ax.set_title("Left-click/drag to PAINT (green). Right-click or 'Erase' to ERASE.")
#         self.ax.set_axis_off()

#         # Try to disable toolbar/pan/zoom so dragging paints instead of moving the image
#         try:
#             self.fig.canvas.toolbar_visible = False     # ipympl
#             self.fig.canvas.header_visible = False      # ipympl
#             self.fig.canvas.resizable = False           # ipympl
#             self.fig.canvas.capture_scroll = True       # ipympl
#         except Exception:
#             pass
#         try:
#             self.ax.set_navigate(False)  # helps in some backends
#         except Exception:
#             pass

#         # Show base image
#         if self.img.ndim == 2:
#             self.base = self.ax.imshow(self.img, cmap="gray", interpolation="nearest")
#         else:
#             self.base = self.ax.imshow(self.img, interpolation="nearest")

#         # Overlay for predicted positives (green with transparency)
#         self.overlay = self.ax.imshow(self._pred_to_rgba(), interpolation="nearest")

#         # Crosshair cursor
#         self.cursor = Cursor(self.ax, useblit=True, color='w', linewidth=1)

#         # Interaction state
#         self.is_drawing = False
#         self.brush_radius = int(brush_radius)
#         self.erase_mode = False

#         # Event wiring
#         self.cid_press   = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
#         self.cid_move    = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
#         self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
#         self.cid_key     = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

#         # Widgets
#         self.brush = IntSlider(description="Brush", value=self.brush_radius, min=1, max=100, step=1, continuous_update=True)
#         self.erase_toggle = ToggleButton(value=False, description="Erase", tooltip="Toggle erase mode (right-click also erases)")
#         self.reset_btn = Button(description="Reset", button_style="warning")
#         self.save_btn  = Button(description="Save", tooltip="Save predicted mask as PNG")
#         self.path_text = Text(value="predicted_mask.png", description="Save as:", layout={'width': '40%'})
#         self.metrics_html = HTML()
#         self.info_html = HTML(value="<b>Black pixels</b> in the ORIGINAL image are ground-truth <b>positive</b>. Painted green = predicted positive.")

#         self.brush.observe(self.on_brush_change, names='value')
#         self.erase_toggle.observe(self.on_erase_toggle, names='value')
#         self.reset_btn.on_click(self.on_reset)
#         self.save_btn.on_click(self.on_save)

#         self.update_metrics()
#         controls = HBox([self.brush, self.erase_toggle, self.reset_btn, self.save_btn, self.path_text])
#         display(VBox([controls, self.metrics_html, self.info_html, out]))

#         # Final render
#         plt.show()

#         # Warn if backend isn't interactive
#         try:
#             # In non-interactive backends, canvas may lack these attributes
#             _ = self.fig.canvas.manager
#         except Exception:
#             print("⚠️ If clicks don't work, enable an interactive backend:\n"
#                   "   JupyterLab: pip install ipympl && use %matplotlib widget\n"
#                   "   Classic Notebook: use %matplotlib notebook")

#     def _pred_to_rgba(self, alpha=100):
#         rgba = np.zeros((self.h, self.w, 4), dtype=np.uint8)
#         rgba[..., 1] = np.where(self.pred, 255, 0)   # Green channel
#         rgba[..., 3] = np.where(self.pred, alpha, 0) # Alpha channel
#         return rgba

#     def update_overlay(self):
#         self.overlay.set_data(self._pred_to_rgba())
#         self.fig.canvas.draw_idle()

#     def on_brush_change(self, change):
#         self.brush_radius = int(change['new'])

#     def on_erase_toggle(self, change):
#         self.erase_mode = bool(change['new'])

#     def on_press(self, event):
#         if event.inaxes != self.ax:
#             return
#         self.is_drawing = True
#         if event.button == 1:      # left click paints
#             self._paint(event.xdata, event.ydata, value=not self.erase_mode)
#         elif event.button == 3:    # right click erases
#             self._paint(event.xdata, event.ydata, value=False)

#     def on_move(self, event):
#         if not self.is_drawing or event.inaxes != self.ax:
#             return
#         if event.button == 1:
#             self._paint(event.xdata, event.ydata, value=not self.erase_mode)
#         elif event.button == 3:
#             self._paint(event.xdata, event.ydata, value=False)

#     def on_release(self, event):
#         self.is_drawing = False

#     def on_key(self, event):
#         if event.key == 'e':
#             self.erase_mode = not self.erase_mode
#             self.erase_toggle.value = self.erase_mode
#         elif event.key == 'r':
#             self.on_reset(None)

#     def _paint(self, x, y, value=True):
#         if x is None or y is None:
#             return
#         cx, cy = int(round(x)), int(round(y))
#         r = max(1, int(self.brush_radius))

#         x0, x1 = max(0, cx - r), min(self.w - 1, cx + r)
#         y0, y1 = max(0, cy - r), min(self.h - 1, cy + r)

#         yy, xx = np.ogrid[y0:y1+1, x0:x1+1]
#         circle = (xx - cx)**2 + (yy - cy)**2 <= r*r

#         self.pred[y0:y1+1, x0:x1+1][circle] = value

#         self.update_overlay()
#         self.update_metrics()

#     def update_metrics(self):
#         stats = metrics_from_masks(self.pred, self.gt)
#         html = f"""
#         <style>
#         .metric-table td, .metric-table th {{ padding: 4px 8px; border-bottom: 1px solid #ddd; }}
#         .metric-table {{ border-collapse: collapse; }}
#         </style>
#         <table class="metric-table">
#           <tr><th align="left">TP</th><td>{stats['TP']}</td>
#               <th align="left">FP</th><td>{stats['FP']}</td>
#               <th align="left">TN</th><td>{stats['TN']}</td>
#               <th align="left">FN</th><td>{stats['FN']}</td>
#               <th align="left">Total</th><td>{stats['Total']}</td></tr>
#           <tr><th align="left">Sensitivity</th><td>{fmt_pct(stats['Sensitivity'])}</td>
#               <th align="left">Specificity</th><td>{fmt_pct(stats['Specificity'])}</td>
#               <th align="left">Precision</th><td>{fmt_pct(stats['Precision'])}</td>
#               <th align="left">NPV</th><td>{fmt_pct(stats['NPV'])}</td>
#               <th align="left">Accuracy</th><td>{fmt_pct(stats['Accuracy'])}</td></tr>
#         </table>
#         """
#         self.metrics_html.value = html

#     def on_reset(self, _):
#         self.pred[:] = False
#         self.update_overlay()
#         self.update_metrics()

#     def on_save(self, _):
#         filename = self.path_text.value.strip() or "predicted_mask.png"
#         arr = (self.pred.astype(np.uint8)) * 255  # 0/255 grayscale
#         Image.fromarray(arr, mode='L').save(filename)
#         with out:
#             clear_output(wait=True)
#             print(f"Saved predicted mask to: {os.path.abspath(filename)}")

# widgets/precision_recall_paint.py

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import HBox, VBox, Button, IntSlider, ToggleButton, HTML, Text, Output, Layout
from IPython.display import display, clear_output
from PIL import Image
import os

# --- Backend check ---
from IPython import get_ipython
_ip = get_ipython()
if _ip:
    try:
        import ipympl
        _ip.run_line_magic("matplotlib", "widget")
    except ImportError:
        pass

# ==== Helpers ====
def load_image(path):
    try:
        img = Image.open(path)
    except Exception as e:
        raise RuntimeError(f"Error loading {path}: {e}")
    return np.array(img.convert("RGB"))

def load_label(path):
    try:
        img = Image.open(path)
    except Exception as e:
        raise RuntimeError(f"Error loading {path}: {e}")
    # Return 2D array
    return np.array(img)[:, :, 1]# Use Grayscale

def compute_gt_mask(img_arr, threshold=10):
    """
    Standard Logic: Pixels > threshold (White) are POSITIVE (Contrails).
    Pixels <= threshold (Black) are NEGATIVE (Background).
    """
    return img_arr > threshold

def metrics_from_masks(pred_pos: np.ndarray, gt_pos: np.ndarray):
    # Fast boolean sums
    tp = np.count_nonzero(pred_pos & gt_pos)
    fp = np.count_nonzero(pred_pos & ~gt_pos)
    tn = np.count_nonzero(~pred_pos & ~gt_pos)
    fn = np.count_nonzero(~pred_pos & gt_pos)
    
    total = tp + fp + tn + fn
    def safe_div(n, d): return float(n)/d if d else 0.0
    
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn, "Total": total,
        "Sensitivity": safe_div(tp, tp + fn),   # Recall
        "Specificity": safe_div(tn, tn + fp),
        "Precision":   safe_div(tp, tp + fp),
        "NPV":         safe_div(tn, tn + fn),
        "Accuracy":    safe_div(tp + tn, total),
        "IOU":         safe_div(tp, tp + fp + fn)
    }

def fmt_pct(x): return f"{x*100:.2f}%"

# ==== Main Class ====
class Painter:
    def __init__(self, image_path, label_path, brush_radius=5):
        self.img = load_image(image_path)
        self.h, self.w = self.img.shape[:2]
        
        # Ground Truth
        raw_label = load_label(label_path)
        self.gt = compute_gt_mask(raw_label, threshold=127) 

        # Prediction State
        self.pred = np.zeros((self.h, self.w), dtype=bool)
        
        # Visual State (RGBA Overlay)
        self.rgba_map = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        self.rgba_map[..., 1] = 255 # Green channel
        self.overlay_alpha = 100
        
        # Setup Figure: 1 Row, 2 Columns (Paint | Ground Truth)
        # We make it wider to accommodate both
        self.fig, (self.ax_paint, self.ax_gt) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Cleanup UI
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False
        self.ax_paint.set_axis_off()
        self.ax_gt.set_axis_off()
        
        # --- LEFT PANEL: Interactive Painting ---
        self.ax_paint.set_title("Paint Here (Green)", fontsize=10)
        self.ax_paint.imshow(self.img, interpolation="nearest")
        self.overlay = self.ax_paint.imshow(self.rgba_map, interpolation="nearest", vmin=0, vmax=255)
        
        # --- RIGHT PANEL: Ground Truth Reference ---
        self.ax_gt.set_title("Ground Truth (Target)", fontsize=10)
        self.ax_gt.imshow(self.gt, cmap="gray", interpolation="nearest")
        
        # Interaction State
        self.brush_radius = brush_radius
        self.erase_mode = False
        self.is_drawing = False
        self.brush_mask = None 
        self._update_brush_mask()

        # Connect Events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        # GUI Controls
        self._init_widgets()
        
        # Initial Draw
        self.update_metrics()
        plt.tight_layout()

    def _init_widgets(self):
        self.out_log = Output()
        
        self.sl_brush = IntSlider(description="Brush", value=self.brush_radius, min=1, max=50)
        self.btn_erase = ToggleButton(description="Erase", icon='eraser')
        self.btn_reset = Button(description="Reset", button_style='warning')
        self.btn_save = Button(description="Save PNG", icon='save')
        self.txt_path = Text(value="predicted_mask.png", layout=Layout(width='150px'))
        self.html_metrics = HTML()
        
        self.sl_brush.observe(self.on_brush_change, names='value')
        self.btn_erase.observe(self.on_erase_toggle, names='value')
        self.btn_reset.on_click(self.on_reset)
        self.btn_save.on_click(self.on_save)

        center_layout = Layout(display='flex', justify_content='center', align_items='center')
        ctrl_box = HBox([self.sl_brush, self.btn_erase, self.btn_reset, self.txt_path, self.btn_save], layout=center_layout)
        
        self.widget_box = VBox([
            ctrl_box,
            self.html_metrics,
            self.fig.canvas,
            self.out_log
        ], layout=center_layout)
        
        display(self.widget_box)

    def _update_brush_mask(self):
        r = self.brush_radius
        y, x = np.ogrid[-r:r+1, -r:r+1]
        self.brush_mask = x**2 + y**2 <= r**2

    def on_brush_change(self, change):
        self.brush_radius = change['new']
        self._update_brush_mask()

    def on_erase_toggle(self, change):
        self.erase_mode = change['new']

    def on_press(self, event):
        if self.fig.canvas.manager is not None and self.fig.canvas.manager.toolbar.mode != '':
            return      
        # Only allow painting on the LEFT axis (ax_paint)
        if event.inaxes != self.ax_paint: return
        self.is_drawing = True
        self._paint(event)

    def on_move(self, event):
        if self.fig.canvas.manager is not None and self.fig.canvas.manager.toolbar.mode != '':
            return    
        if not self.is_drawing or event.inaxes != self.ax_paint: return
        self._paint(event)

    def on_release(self, event):
        self.is_drawing = False
        self.update_metrics()

    def _paint(self, event):
        cx, cy = int(event.xdata), int(event.ydata)
        r = self.brush_radius
        
        y_min, y_max = max(0, cy - r), min(self.h, cy + r + 1)
        x_min, x_max = max(0, cx - r), min(self.w, cx + r + 1)
        
        mask_y_min = r - (cy - y_min)
        mask_y_max = mask_y_min + (y_max - y_min)
        mask_x_min = r - (cx - x_min)
        mask_x_max = mask_x_min + (x_max - x_min)
        
        local_mask = self.brush_mask[mask_y_min:mask_y_max, mask_x_min:mask_x_max]
        
        target_val = not self.erase_mode
        current_slice = self.pred[y_min:y_max, x_min:x_max]
        current_slice[local_mask] = target_val
        
        alpha_slice = self.rgba_map[y_min:y_max, x_min:x_max, 3]
        alpha_slice[local_mask] = self.overlay_alpha if target_val else 0
        
        self.overlay.set_data(self.rgba_map)
        self.fig.canvas.draw_idle()

    def update_metrics(self):
        stats = metrics_from_masks(self.pred, self.gt)
        
        html_content = f"""
        <div style="display: flex; justify-content: center; font-family: sans-serif; margin-bottom: 10px;">
        <table style="border-collapse: collapse; width: 650px; text-align: left; font-size: 14px;">
          <tr style="border-bottom: 2px solid #aaa; background-color: #f4f4f4;">
            <th style="padding: 6px;">Metric</th>
            <th style="padding: 6px;">Precision</th>
            <th style="padding: 6px;">Recall</th>
            <th style="padding: 6px;">IOU</th>
            <th style="padding: 6px;">Accuracy</th>
          </tr>
          <tr>
            <td style="padding: 6px;"><b>Current</b></td>
            <td style="padding: 6px; font-weight: bold; color: {'green' if stats['Precision'] > 0.5 else '#d9534f'}">{fmt_pct(stats['Precision'])}</td>
            <td style="padding: 6px; font-weight: bold; color: {'green' if stats['Sensitivity'] > 0.5 else '#d9534f'}">{fmt_pct(stats['Sensitivity'])}</td>
            <td style="padding: 6px;">{fmt_pct(stats['IOU'])}</td>
            <td style="padding: 6px;">{fmt_pct(stats['Accuracy'])}</td>
          </tr>
           <tr style="font-size: 0.85em; color: #666; border-top: 1px solid #eee;">
             <td colspan="5" style="padding: 4px; text-align: center;">
               True Pos: {stats['TP']} | False Pos: {stats['FP']} | False Neg: {stats['FN']}
             </td>
          </tr>
        </table>
        </div>
        """
        self.html_metrics.value = html_content

    def on_reset(self, _):
        self.pred.fill(False)
        self.rgba_map[..., 3] = 0
        self.overlay.set_data(self.rgba_map)
        self.fig.canvas.draw_idle()
        self.update_metrics()

    def on_save(self, _):
        path = self.txt_path.value
        to_save = (self.pred.astype(np.uint8) * 255)
        try:
            Image.fromarray(to_save).save(path)
            with self.out_log:
                clear_output(wait=True)
                print(f"Saved to {os.path.abspath(path)}")
        except Exception as e:
             with self.out_log:
                print(f"Error saving: {e}")