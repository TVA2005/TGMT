# Copilot Instructions for TGMT

## Project Overview
**TGMT** is a computer vision learning project organized into multiple subdirectories, with the main active development in `cv-day2-RBG/`. The project focuses on **image processing with OpenCV, NumPy visualization, and interactive graphics**.

**Key Stack:**
- **OpenCV (cv2)** - Core image processing library
- **NumPy** - Numerical arrays and image manipulation
- **Matplotlib** - Visualization and plotting
- **Requests/urllib** - Downloading images from URLs

## Architecture & Components

### Directory Structure
- **Root:** `main.py` (stub - "Hello from tgmt!")
- **`cv-day2-RBG/`** (Active development)
  - `main.py` - **Grayscale/RGB gradient generation** and clock visualization with matplotlib
  - `main1.py` - **Comprehensive image creation demos** (617 lines) - includes cat image loading, color space conversions, pixel manipulation
  - `main2.py` - **3D visualization** (534 lines) - rotating Earth, 3D plots with mpl_toolkits
  - `main3.py` - **Image processing with noise** - downloads images from GitHub, applies Gaussian noise, concatenates results

### Data Flow Patterns
1. **Image Source → Processing → Display**
   - Images created programmatically (NumPy arrays) OR fetched from URLs
   - Processing applied (noise, gradients, transformations)
   - Displayed via OpenCV (`cv2.imshow`) or Matplotlib (`plt.imshow`)

2. **Color Space Handling**
   - **OpenCV uses BGR format** (not RGB) - use `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)` when converting from matplotlib
   - Matplotlib uses RGB format

3. **Image Data Types**
   - Always use `dtype=np.uint8` for 8-bit image arrays (0-255 pixel values)
   - Clip values when adding noise: `np.clip(arr, 0, 255).astype(np.uint8)`

## Critical Developer Patterns

### Image Display Windows (OpenCV)
```python
cv2.imshow("Window Title", img)
cv2.waitKey(0)          # 0 = wait indefinitely, 1000 = 1 second
cv2.destroyAllWindows()
```
- Used extensively in all main*.py files
- Always call `destroyAllWindows()` to clean up resources

### Remote Image Loading (main3.py pattern)
```python
import urllib.request as request
resp = request.urlopen(url)
arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
img = cv.imdecode(arr, cv.IMREAD_COLOR)
```
- Primary method for GitHub-hosted test images

### Array Concatenation for Comparison Views
```python
compile_img = np.concatenate((img, img2), axis=1)  # Horizontal
# axis=0 for vertical stacking
```

### Matplotlib Interactive Mode
```python
plt.ion()        # Enable interactive mode for real-time updates
fig, ax = plt.subplots()
ax.clear()       # Clear before redrawing (used in real-time loops)
plt.show()
```
- Used for clock visualization (main.py) and Earth rotation (main2.py)

## Integration Points

### External Dependencies
- `numpy`, `opencv-python`, `matplotlib`, `pillow`, `requests` defined in [pyproject.toml](cv-day2-RBG/pyproject.toml)
- No external APIs - all data is either procedurally generated or fetched from public GitHub repos

### Cross-Component Communication
- **main3.py** is self-contained; reads from GitHub, processes locally
- **main1.py** has modular functions (e.g., `load_cat_image()`) but no imports from other local modules
- No shared utilities library - each file is independent

## Conventions & Common Tasks

### Running Scripts
```powershell
python cv-day2-RBG/main3.py   # Works on Windows (cv2.imshow requires GUI)
```

### Debugging Tips
1. Commented-out display blocks show intended visualization points
2. Use `print()` for pixel value inspection during development
3. Check array shapes with `.shape` attribute when concatenating images

### When Adding New Image Processing
- Test with small images first (performance can be slow with 1024×768 arrays)
- Always validate dtype as `uint8` before display
- Use `np.random.seed()` for reproducible noise in testing

## Python Version & Environment
- Requires Python ≥ 3.14 (per pyproject.toml)
- No virtual environment explicitly set up - install dependencies via pip

---

**Last Updated:** 2026-01-28
