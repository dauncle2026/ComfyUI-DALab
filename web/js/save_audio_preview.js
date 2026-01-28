import { app } from '../../../scripts/app.js';
import { api } from '../../../scripts/api.js';

const MARGIN = 10;
let stylesInjected = false;

function ensureStyles() {
  if (stylesInjected) return;
  stylesInjected = true;
  const style = document.createElement('style');
  style.textContent = `
    .feishu-audio-gallery {
      position: absolute;
      z-index: 1;
      pointer-events: auto;
      
      display: flex;
      flex-direction: column; 
      gap: 6px;
      
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
    
    .feishu-audio-gallery[data-state="active"] {
      opacity: 1;
      pointer-events: auto;
    }

    .feishu-audio-item {
      display: flex;
      flex-direction: column;
      
      width: 100%;
      min-width: 0; 
      
      background: rgba(0, 0, 0, 0.5);
      padding: 6px;
      border-radius: 4px;
      border: 1px solid rgba(255,255,255,0.1);
      box-sizing: border-box;
    }

    .feishu-audio-title {
      font-size: 11px;
      color: #ccc;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-bottom: 4px;
      padding-left: 2px;
    }

    .feishu-audio-player {
      width: 100%;
      height: 32px;
      display: block;
      margin-top: 2px;
    }
  `;
  document.head.appendChild(style);
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

function buildAudioUrl(item) {
  if (!item || !item.filename) return null;
  const params = new URLSearchParams({
    filename: item.filename,
    type: item.type || 'output',
    format: item.format || 'flac'
  });
  if (item.subfolder) params.set('subfolder', item.subfolder);
  
  let url = api.apiURL(`/view?${params.toString()}`);
  if (typeof app.getRandParam === 'function') url += app.getRandParam();
  return url;
}

function renderAudiosToContainer(container, audioList) {
  container.innerHTML = '';

  if (!audioList || audioList.length === 0) {
    container.dataset.state = 'inactive';
    return;
  }

  container.dataset.state = 'active';

  audioList.forEach((item) => {
    const url = buildAudioUrl(item);
    if (!url) return;

    const itemDiv = document.createElement('div');
    itemDiv.className = 'feishu-audio-item';

    const title = document.createElement('div');
    title.className = 'feishu-audio-title';
    title.title = item.filename;
    title.textContent = item.filename;

    const audio = document.createElement('audio');
    audio.className = 'feishu-audio-player';
    
    audio.controls = true;
    audio.preload = "metadata"; 
    audio.src = url;

    itemDiv.appendChild(title);
    itemDiv.appendChild(audio);
    container.appendChild(itemDiv);
  });
}

// 获取或创建节点的audio container
function getOrCreateContainer(node) {
  const containerId = `feishu-audio-gallery-${node.id}`;
  let container = document.getElementById(containerId);
  
  if (!container) {
    container = document.createElement('div');
    container.id = containerId;
    container.className = 'feishu-audio-gallery';
    container.dataset.state = 'inactive';
    document.body.appendChild(container);
  }
  
  return container;
}

// 获取节点存储的音频数据
function getNodeAudios(node) {
  // 优先从properties获取（持久化存储，tab切换后仍然存在）
  if (node.properties?.savedAudios && node.properties.savedAudios.length > 0) {
    return node.properties.savedAudios;
  }
  // 其次从audios属性获取
  if (node.audios && Array.isArray(node.audios) && node.audios.length > 0) {
    return node.audios;
  }
  return null;
}

// 设置节点的音频数据
function setNodeAudios(node, audios) {
  // 存储到audios属性
  node.audios = audios;
  // 同时存储到properties（用于序列化持久化）
  if (!node.properties) {
    node.properties = {};
  }
  node.properties.savedAudios = audios;
}

app.registerExtension({
  name: 'Dauncle.DASaveAudioPreview', 
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== 'DASaveAudio') return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      originalOnNodeCreated?.apply(this, args);

      ensureStyles();

      // 初始化properties（用于序列化持久化）
      if (!this.properties) {
        this.properties = {};
      }
      if (!this.properties.hasOwnProperty('savedAudios')) {
        this.properties.savedAudios = [];
      }
      this.lastDomHeight = 0;

      const widget = {
        name: 'feishu_audio_gallery_widget',
        type: 'feishu_audio_gallery_widget',
        
        draw(ctx, node, widgetWidth, y) {
          const container = getOrCreateContainer(node);
          
          // 检查是否需要恢复音频数据
          const hasAudiosInContainer = container.querySelectorAll('.feishu-audio-item').length > 0;
          const storedAudios = getNodeAudios(node);
          
          // 如果container是空的但节点有存储的音频数据，恢复它
          if (!hasAudiosInContainer && storedAudios && storedAudios.length > 0) {
            renderAudiosToContainer(container, storedAudios);
          }
          
          const targetWidth = (node.size?.[0] ?? widgetWidth) - MARGIN * 2;
          container.style.width = `${targetWidth}px`;
          
          const style = buildTransformStyle(ctx, widgetWidth, y);
          Object.assign(container.style, style);
          
          const currentHeight = container.offsetHeight;
          
          if (container.dataset.state === 'active' && currentHeight > 0) {
             const requiredHeight = y + currentHeight + MARGIN * 2;
             
             if (node.size[1] < requiredHeight - 1 || Math.abs(node.size[1] - requiredHeight) > 10) {
                 node.setSize([node.size[0], requiredHeight]);
                 node.setDirtyCanvas(true, true); 
             }
          }
        },
        computeSize() { return [200, 10]; }, 
      };

      this.addCustomWidget(widget);
      this.size = [Math.max(this.size[0], 280), this.size[1]];

      const originalOnRemoved = this.onRemoved;
      this.onRemoved = function () {
        const containerId = `feishu-audio-gallery-${this.id}`;
        const container = document.getElementById(containerId);
        if (container) {
          container.remove();
        }
        originalOnRemoved?.apply(this, arguments);
      };

      const originalOnExecuted = this.onExecuted;
      this.onExecuted = function (message) {
        originalOnExecuted?.apply(this, arguments);
        const audioResults = message?.custom_audios || [];
        if (Array.isArray(audioResults) && audioResults.length > 0) {
          // 存储音频数据
          setNodeAudios(this, audioResults);
          // 更新gallery
          const container = getOrCreateContainer(this);
          renderAudiosToContainer(container, audioResults);
        }
      };
    };
  },
});
