import { app } from '../../../scripts/app.js';
import { api } from '../../../scripts/api.js';

const MARGIN = 10;
const GAP = 6;
const MIN_ITEM_WIDTH = 150;  

const COLUMN_RULES = [
  [2, 2], 
  [4, 2],
  [6, 3],
  [9, 3],
  [Infinity, 4],
];

let stylesInjected = false;

function ensureStyles() {
  if (stylesInjected) return;
  stylesInjected = true;
  const style = document.createElement('style');
  style.textContent = `
    .da-image-gallery {
      position: absolute;
      z-index: 999;
      
      display: grid;
      gap: ${GAP}px;
      justify-content: start;
      align-content: start;
      
      background: transparent;
      border: none;
      box-shadow: none;
      padding: 0;
      
      height: auto;
      overflow: visible;
      
      transition: opacity 0.2s ease;
      opacity: 0;
      pointer-events: none;
      
      box-sizing: border-box;
      padding-bottom: 10px;
    }
    
    .da-image-gallery[data-state="active"] {
      opacity: 1;
      pointer-events: auto;
    }

    .da-image-item {
      display: flex;
      flex-direction: column;
      
      width: 100%;
      min-width: 0;
      
      background: rgba(0, 0, 0, 0.4);
      padding: 3px;
      border-radius: 4px;
      border: 1px solid rgba(255,255,255,0.1);
      box-sizing: border-box;
      
      height: fit-content;
      cursor: pointer;
      transition: border-color 0.15s ease, transform 0.15s ease;
    }

    .da-image-item:hover {
      border-color: rgba(255,255,255,0.3);
      transform: scale(1.02);
    }

    .da-image-wrapper {
      width: 100%;
      position: relative;
      background: #111;
      border-radius: 2px;
      overflow: hidden;
    }

    .da-image-preview {
      width: 100%;
      height: 100%;
      display: block;
      object-fit: contain;
    }

    .da-image-title {
      font-size: 9px;
      color: #aaa;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-top: 3px;
      text-align: center;
      padding: 0 2px;
    }

    /* 图片放大查看的遮罩层 */
    .da-image-lightbox {
      position: fixed;
      top: 0;
      left: 0;
      width: 100vw;
      height: 100vh;
      background: rgba(0, 0, 0, 0.9);
      z-index: 10000;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: zoom-out;
    }

    .da-image-lightbox img {
      max-width: 95vw;
      max-height: 95vh;
      object-fit: contain;
      border-radius: 4px;
      box-shadow: 0 4px 30px rgba(0,0,0,0.5);
    }
  `;
  document.head.appendChild(style);
}

function getMaxColumns(imageCount) {
  for (const [threshold, maxCols] of COLUMN_RULES) {
    if (imageCount <= threshold) {
      return maxCols;
    }
  }
  return COLUMN_RULES[COLUMN_RULES.length - 1][1];
}

function calcColumns(containerWidth, imageCount) {
  const maxColumns = getMaxColumns(imageCount);
  let cols = Math.floor((containerWidth + GAP) / (MIN_ITEM_WIDTH + GAP));
  cols = Math.max(1, Math.min(cols, maxColumns, imageCount));
  return cols;
}

function buildTransformStyle(ctx, widgetWidth, y) {
  const { canvas } = ctx;
  const rect = canvas.getBoundingClientRect();
  const matrix = new DOMMatrix()
    .scaleSelf(rect.width / canvas.width, rect.height / canvas.height)
    .multiplySelf(ctx.getTransform())
    .translateSelf(MARGIN, y + MARGIN);
  
  return {
    transformOrigin: '0 0',
    transform: matrix.toString(),
    left: `${rect.left + window.scrollX}px`,
    top: `${rect.top + window.scrollY}px`,
  };
}

function buildImageUrl(item) {
  if (!item || !item.filename) return null;
  const params = new URLSearchParams({
    filename: item.filename,
    type: item.type || 'output',
  });
  if (item.subfolder) params.set('subfolder', item.subfolder);
  
  let url = api.apiURL(`/view?${params.toString()}`);
  if (typeof app.getRandParam === 'function') url += app.getRandParam();
  return url;
}

// 显示图片放大查看
function showLightbox(imageUrl) {
  const lightbox = document.createElement('div');
  lightbox.className = 'da-image-lightbox';
  
  const img = document.createElement('img');
  img.src = imageUrl;
  
  lightbox.appendChild(img);
  document.body.appendChild(lightbox);
  
  lightbox.addEventListener('click', () => {
    lightbox.remove();
  });
  
  // ESC键关闭
  const handleKeydown = (e) => {
    if (e.key === 'Escape') {
      lightbox.remove();
      document.removeEventListener('keydown', handleKeydown);
    }
  };
  document.addEventListener('keydown', handleKeydown);
}

function renderImagesToContainer(container, imageList) {
  container.innerHTML = '';

  if (!imageList || imageList.length === 0) {
    container.dataset.state = 'inactive';
    return;
  }

  container.dataset.state = 'active';

  imageList.forEach((item) => {
    const url = buildImageUrl(item);
    if (!url) return;

    const itemDiv = document.createElement('div');
    itemDiv.className = 'da-image-item';

    const imageWrapper = document.createElement('div');
    imageWrapper.className = 'da-image-wrapper';

    // 设置宽高比（默认1:1，如果有宽高信息则使用）
    let ratio = "1 / 1";
    if (item.width && item.height) {
      ratio = `${item.width} / ${item.height}`;
    }
    imageWrapper.style.aspectRatio = ratio;

    const img = document.createElement('img');
    img.className = 'da-image-preview';
    img.src = url;
    img.loading = 'lazy';
    
    // 图片加载后更新宽高比
    img.onload = function() {
      if (!item.width || !item.height) {
        imageWrapper.style.aspectRatio = `${img.naturalWidth} / ${img.naturalHeight}`;
      }
    };

    // 点击放大查看
    itemDiv.addEventListener('click', () => {
      showLightbox(url);
    });

    const title = document.createElement('div');
    title.className = 'da-image-title';
    title.title = item.filename;
    title.textContent = item.filename.replace(/\.[^/.]+$/, ''); // 移除扩展名

    imageWrapper.appendChild(img);
    itemDiv.appendChild(imageWrapper);
    itemDiv.appendChild(title);
    container.appendChild(itemDiv);
  });
}

// 获取或创建节点的image container
function getOrCreateContainer(node) {
  const containerId = `da-image-gallery-${node.id}`;
  let container = document.getElementById(containerId);
  
  if (!container) {
    container = document.createElement('div');
    container.id = containerId;
    container.className = 'da-image-gallery';
    container.dataset.state = 'inactive';
    document.body.appendChild(container);
  }
  
  return container;
}

// 获取节点存储的图片数据
function getNodeImages(node) {
  // 优先从properties获取（持久化存储，刷新后仍然存在）
  if (node.properties?.savedImages && node.properties.savedImages.length > 0) {
    return node.properties.savedImages;
  }
  return null;
}

// 设置节点的图片数据
function setNodeImages(node, images) {
  // 存储到properties（用于序列化持久化）
  if (!node.properties) {
    node.properties = {};
  }
  node.properties.savedImages = images;
}

app.registerExtension({
  name: 'Dauncle.DASaveImagePreview',
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== 'DASaveImage========') return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      originalOnNodeCreated?.apply(this, args);

      ensureStyles();

      // 初始化properties（用于序列化持久化）
      if (!this.properties) {
        this.properties = {};
      }
      if (!this.properties.hasOwnProperty('savedImages')) {
        this.properties.savedImages = [];
      }
      this.lastDomHeight = 0;

      const widget = {
        name: 'da_image_gallery_widget',
        type: 'da_image_gallery_widget',
        
        draw(ctx, node, widgetWidth, y) {
          const container = getOrCreateContainer(node);
          
          // 检查是否需要恢复图片数据
          const hasImagesInContainer = container.querySelectorAll('.da-image-item').length > 0;
          const storedImages = getNodeImages(node);
          
          // 如果container是空的但节点有存储的图片数据，恢复它
          if (!hasImagesInContainer && storedImages && storedImages.length > 0) {
            renderImagesToContainer(container, storedImages);
          }
          
          const targetWidth = (node.size?.[0] ?? widgetWidth) - MARGIN * 2;
          container.style.width = `${targetWidth}px`;
          
          const imageCount = container.querySelectorAll('.da-image-item').length || 1;
          const cols = calcColumns(targetWidth, imageCount);
          container.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
          
          const style = buildTransformStyle(ctx, widgetWidth, y);
          Object.assign(container.style, style);
          
          if (container.dataset.state === 'active') {
            requestAnimationFrame(() => {
              const currentHeight = container.offsetHeight;
              if (currentHeight > 0) {
                const requiredHeight = y + currentHeight + MARGIN * 2;
                const currentNodeHeight = node.size[1];
                
                if (Math.abs(currentNodeHeight - requiredHeight) > 5) {
                  node.setSize([node.size[0], requiredHeight]);
                  node.setDirtyCanvas(true, true);
                }
              }
            });
          }
        },
        
        computeSize() { 
          return [MIN_ITEM_WIDTH * 2 + GAP + MARGIN * 2, 10]; 
        }, 
      };

      this.addCustomWidget(widget);
      const defaultWidth = MIN_ITEM_WIDTH * 2 + GAP + MARGIN * 2 + 20;
      this.size = [Math.max(this.size[0], defaultWidth), this.size[1]];

      const originalOnRemoved = this.onRemoved;
      this.onRemoved = function () {
        const containerId = `da-image-gallery-${this.id}`;
        const container = document.getElementById(containerId);
        if (container) {
          container.remove();
        }
        originalOnRemoved?.apply(this, arguments);
      };

      const originalOnExecuted = this.onExecuted;
      this.onExecuted = function (message) {
        originalOnExecuted?.apply(this, arguments);
        // ComfyUI 标准的 SavedImages 返回格式是 { images: [...] }
        const imageResults = message?.images || [];
        if (Array.isArray(imageResults) && imageResults.length > 0) {
          // 存储图片数据
          setNodeImages(this, imageResults);
          // 更新gallery
          const container = getOrCreateContainer(this);
          renderImagesToContainer(container, imageResults);
        }
      };
    };
  },
});
