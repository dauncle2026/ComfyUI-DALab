import { app } from '../../../scripts/app.js';
import { api } from '../../../scripts/api.js';

const MARGIN = 10;
const GAP = 8;
const MIN_ITEM_WIDTH = 200;  

const COLUMN_RULES = [
  [2, 2], 
  [6, 3],
  [8, 4],
  [Infinity, 5],
];

let stylesInjected = false;

function ensureStyles() {
  if (stylesInjected) return;
  stylesInjected = true;
  const style = document.createElement('style');
  style.textContent = `
    .feishu-video-gallery {
      position: absolute;
      z-index: 1;
      
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
    
    .feishu-video-gallery[data-state="active"] {
      opacity: 1;
      pointer-events: auto;
    }

    .feishu-video-item {
      display: flex;
      flex-direction: column;
      
      width: 100%;
      min-width: 0;
      
      background: rgba(0, 0, 0, 0.6);
      padding: 4px;
      border-radius: 4px;
      border: 1px solid rgba(255,255,255,0.15);
      box-sizing: border-box;
      
      height: fit-content;
    }

    .feishu-video-title {
      font-size: 10px;
      color: #ddd;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-bottom: 2px;
      text-align: center;
      padding: 0 2px;
    }

    .feishu-video-size {
      font-size: 9px;
      color: #888;
      text-align: center;
      margin-bottom: 4px;
    }

    .feishu-video-wrapper {
      width: 100%;
      position: relative;
      background: #000;
      border-radius: 2px;
      overflow: hidden;
    }

    .feishu-video-player {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      display: block;
      object-fit: contain; 
    }
    
    .feishu-video-player::-webkit-media-controls {
      transform: scale(0.85);
      transform-origin: bottom center;
    }
  `;
  document.head.appendChild(style);
}

function getMaxColumns(videoCount) {
  for (const [threshold, maxCols] of COLUMN_RULES) {
    if (videoCount <= threshold) {
      return maxCols;
    }
  }
  return COLUMN_RULES[COLUMN_RULES.length - 1][1];
}

function calcColumns(containerWidth, videoCount) {
  const maxColumns = getMaxColumns(videoCount);
  let cols = Math.floor((containerWidth + GAP) / (MIN_ITEM_WIDTH + GAP));
  cols = Math.max(1, Math.min(cols, maxColumns, videoCount));
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

function buildVideoUrl(item) {
  if (!item || !item.filename) return null;
  const params = new URLSearchParams({
    filename: item.filename,
    type: item.type || 'output',
    format: item.format || 'mp4'
  });
  if (item.subfolder) params.set('subfolder', item.subfolder);
  
  let url = api.apiURL(`/view?${params.toString()}`);
  if (typeof app.getRandParam === 'function') url += app.getRandParam();
  return url;
}

function renderVideosToContainer(container, videoList) {
  container.innerHTML = '';

  if (!videoList || videoList.length === 0) {
    container.dataset.state = 'inactive';
    return;
  }

  container.dataset.state = 'active';

  videoList.forEach((item) => {
    const url = buildVideoUrl(item);
    if (!url) return;

    const itemDiv = document.createElement('div');
    itemDiv.className = 'feishu-video-item';

    const title = document.createElement('div');
    title.className = 'feishu-video-title';
    title.title = item.filename;
    title.textContent = item.filename.replace('.mp4', '');

    // 显示视频尺寸
    const sizeInfo = document.createElement('div');
    sizeInfo.className = 'feishu-video-size';
    if (item.width && item.height) {
      sizeInfo.textContent = `${item.width} × ${item.height}`;
    } else {
      sizeInfo.textContent = '';
    }

    const videoWrapper = document.createElement('div');
    videoWrapper.className = 'feishu-video-wrapper';

    let ratio = "16 / 9";
    if (item.width && item.height) {
        ratio = `${item.width} / ${item.height}`;
    }
    videoWrapper.style.aspectRatio = ratio;

    const video = document.createElement('video');
    video.className = 'feishu-video-player';
    
    video.controls = true;
    video.loop = true;
    video.muted = true; 
    video.autoplay = true;
    video.src = url;

    videoWrapper.appendChild(video);
    itemDiv.appendChild(title);
    itemDiv.appendChild(sizeInfo);
    itemDiv.appendChild(videoWrapper);
    container.appendChild(itemDiv);
  });
}

// 获取或创建节点的video container
function getOrCreateContainer(node) {
  const containerId = `feishu-video-gallery-${node.id}`;
  let container = document.getElementById(containerId);
  
  if (!container) {
    container = document.createElement('div');
    container.id = containerId;
    container.className = 'feishu-video-gallery';
    container.dataset.state = 'inactive';
    document.body.appendChild(container);
  }
  
  return container;
}

// 获取节点存储的视频数据
function getNodeVideos(node) {
  // 优先从properties获取（持久化存储，tab切换后仍然存在）
  if (node.properties?.savedVideos && node.properties.savedVideos.length > 0) {
    return node.properties.savedVideos;
  }
  // 其次从images属性获取（ComfyUI标准方式）
  if (node.images && Array.isArray(node.images) && node.images.length > 0) {
    return node.images;
  }
  return null;
}

// 设置节点的视频数据
function setNodeVideos(node, videos) {
  // 存储到images属性（ComfyUI标准方式，用于显示）
  node.images = videos;
  // 同时存储到properties（用于序列化持久化）
  if (!node.properties) {
    node.properties = {};
  }
  node.properties.savedVideos = videos;
}

app.registerExtension({
  name: 'Dauncle.DASaveVideoPreview',
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== 'DASaveVideo' && nodeData?.name !== 'DAConcatVideo') return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      originalOnNodeCreated?.apply(this, args);

      ensureStyles();

      // 初始化properties（用于序列化持久化）
      if (!this.properties) {
        this.properties = {};
      }
      if (!this.properties.hasOwnProperty('savedVideos')) {
        this.properties.savedVideos = [];
      }
      this.lastDomHeight = 0;

      const widget = {
        name: 'feishu_video_gallery_widget',
        type: 'feishu_video_gallery_widget',
        
        draw(ctx, node, widgetWidth, y) {
          const container = getOrCreateContainer(node);
          
          // 检查是否需要恢复视频数据
          const hasVideosInContainer = container.querySelectorAll('.feishu-video-item').length > 0;
          const storedVideos = getNodeVideos(node);
          
          // 如果container是空的但节点有存储的视频数据，恢复它
          if (!hasVideosInContainer && storedVideos && storedVideos.length > 0) {
            renderVideosToContainer(container, storedVideos);
          }
          
          const targetWidth = (node.size?.[0] ?? widgetWidth) - MARGIN * 2;
          container.style.width = `${targetWidth}px`;
          
          const videoCount = container.querySelectorAll('.feishu-video-item').length || 1;
          const cols = calcColumns(targetWidth, videoCount);
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
        const containerId = `feishu-video-gallery-${this.id}`;
        const container = document.getElementById(containerId);
        if (container) {
          container.remove();
        }
        originalOnRemoved?.apply(this, arguments);
      };

      const originalOnExecuted = this.onExecuted;
      this.onExecuted = function (message) {
        originalOnExecuted?.apply(this, arguments);
        const videoResults = message?.custom_videos || [];
        if (Array.isArray(videoResults) && videoResults.length > 0) {
          // 存储视频数据
          setNodeVideos(this, videoResults);
          // 更新gallery
          const container = getOrCreateContainer(this);
          renderVideosToContainer(container, videoResults);
        }
      };
    };
  },
});

