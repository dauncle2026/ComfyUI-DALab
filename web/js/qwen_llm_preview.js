import { app } from '../../../scripts/app.js';

let stylesInjected = false;
function ensureStyles() {
    if (stylesInjected) return;
    stylesInjected = true;
    const style = document.createElement('style');
    style.textContent = `
        .qwen-prompt-container {
            position: absolute;
            transform-origin: 0 0;
            
            display: flex;
            flex-direction: column;
            gap: 6px;
            
            /* 限制最大高度，内容过多出滚动条 */
            max-height: 400px; 
            overflow-y: auto;
            overflow-x: hidden;
            
            opacity: 0;
            pointer-events: none; 
            
            box-sizing: border-box;
            font-family: 'Menlo', 'Monaco', 'Consolas', monospace;
            padding-right: 4px;
            padding-bottom: 10px;
        }

        .qwen-prompt-container[data-state="active"] {
            opacity: 1;
            pointer-events: auto;
        }

        .qwen-prompt-container::-webkit-scrollbar { width: 6px; }
        .qwen-prompt-container::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
        .qwen-prompt-container::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }

        .qwen-prompt-card {
            background: #222;
            border-left: 3px solid #4ade80;
            border-radius: 2px;
            padding: 8px 10px; 
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-bottom: 2px;
        }

        .qwen-prompt-header {
            display: flex; 
            justify-content: space-between; 
            align-items: center;
            border-bottom: 1px solid #333;
            padding-bottom: 4px;
            margin-bottom: 2px;
        }

        .qwen-index-tag { 
            font-size: 11px; 
            color: #4ade80; 
            font-weight: bold; 
        }

        .qwen-prompt-en { 
            color: #eee;
            font-size: 13px; 
            line-height: 1.4; 
            white-space: pre-wrap; 
            word-break: break-word;
        }

        .qwen-prompt-zh { 
            color: #888; 
            font-size: 13px; 
            line-height: 1.4;
        }
    `;
    document.head.appendChild(style);
}

function renderPromptsToContainer(container, promptList) {
    container.innerHTML = '';
    if (!promptList || promptList.length === 0) {
        container.dataset.state = 'inactive';
        return 0; 
    }
    container.dataset.state = 'active';

    promptList.forEach((item, index) => {
        const card = document.createElement('div');
        card.className = 'qwen-prompt-card';
        
        const header = document.createElement('div');
        header.className = 'qwen-prompt-header';
        header.innerHTML = `<span class="qwen-index-tag">PROMPT ${String(index + 1).padStart(2, '0')}</span>`;
        
        const enDiv = document.createElement('div');
        enDiv.className = 'qwen-prompt-en';
        enDiv.textContent = item.en_prompt;

        const zhDiv = document.createElement('div');
        zhDiv.className = 'qwen-prompt-zh';
        zhDiv.textContent = item.zh_prompt;

        card.appendChild(header);
        card.appendChild(enDiv);
        card.appendChild(zhDiv);
        container.appendChild(card);
    });

    return container.scrollHeight; 
}

// 获取或创建节点的prompt container
function getOrCreateContainer(node) {
    const containerId = `qwen-prompt-container-${node.id}`;
    let container = document.getElementById(containerId);
    
    if (!container) {
        container = document.createElement('div');
        container.id = containerId;
        container.className = 'qwen-prompt-container';
        document.body.appendChild(container);
    }
    
    return container;
}

// 获取节点存储的prompt数据
function getNodePrompts(node) {
    // 优先从properties获取（持久化存储，tab切换后仍然存在）
    if (node.properties?.savedPrompts && node.properties.savedPrompts.length > 0) {
        return node.properties.savedPrompts;
    }
    return null;
}

// 设置节点的prompt数据
function setNodePrompts(node, prompts, height = null) {
    // 存储到properties（用于序列化持久化）
    if (!node.properties) {
        node.properties = {};
    }
    node.properties.savedPrompts = prompts;
    // 同时保存高度信息
    if (height !== null) {
        node.properties.savedWidgetHeight = height;
    }
}

// 获取保存的widget高度
function getSavedWidgetHeight(node) {
    return node.properties?.savedWidgetHeight || null;
}

app.registerExtension({
    name: 'UncleDa.DAQwenLLMPreview',
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== 'DAQwenLLM' && nodeData?.name !== 'DAQwenVL'){
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            originalOnNodeCreated?.apply(this, args);

            ensureStyles();

            // 初始化properties（用于序列化持久化）
            if (!this.properties) {
                this.properties = {};
            }
            if (!this.properties.hasOwnProperty('savedPrompts')) {
                this.properties.savedPrompts = [];
            }

            // 恢复保存的高度，或使用默认值
            const savedHeight = getSavedWidgetHeight(this);
            this.qwenWidgetHeight = savedHeight || 60; 

            // 保存节点引用供widget使用
            const nodeRef = this;
            
            // 用于防止在同一帧内多次触发重新布局
            let lastRestoredTime = 0;

            const widget = {
                name: 'qwen_prompt_gallery_widget',
                type: 'qwen_prompt_gallery_widget',
                
                // 告诉 ComfyUI 这个 Widget 的真实高度（使用节点实例的高度）
                computeSize(width) { 
                    return [width, nodeRef.qwenWidgetHeight || 60];
                },

                draw(ctx, node, widgetWidth, y) {
                    const container = getOrCreateContainer(node);
                    const hasPromptsInContainer = container.querySelectorAll('.qwen-prompt-card').length > 0;
                    const storedPrompts = getNodePrompts(node);
                    const now = Date.now();
                    
                    // 如果container是空的但有存储的数据，恢复它
                    // 使用时间戳避免频繁触发（至少间隔100ms）
                    if (!hasPromptsInContainer && storedPrompts && storedPrompts.length > 0 && (now - lastRestoredTime > 100)) {
                        const contentHeight = renderPromptsToContainer(container, storedPrompts);
                        lastRestoredTime = now;
                        
                        // 使用保存的高度，或者重新计算
                        const savedH = getSavedWidgetHeight(node);
                        const MAX_HEIGHT = 400;
                        const neededHeight = savedH || (Math.min(contentHeight, MAX_HEIGHT) + 20);
                        node.qwenWidgetHeight = neededHeight;
                        
                        // 延迟触发重新布局
                        setTimeout(() => {
                            if (node.onResize) {
                                node.onResize(node.size);
                            }
                            node.graph?.setDirtyCanvas(true, true);
                        }, 100);
                    }

                    const transform = ctx.getTransform();
                    const rect = ctx.canvas.getBoundingClientRect();
                    const margin = 10;
                    const elWidth = widgetWidth - (margin * 2);
                    
                    const matrix = new DOMMatrix()
                        .scaleSelf(rect.width / ctx.canvas.width, rect.height / ctx.canvas.height)
                        .multiplySelf(transform)
                        .translateSelf(margin, y + margin);

                    container.style.transform = matrix.toString();
                    container.style.width = `${elWidth}px`;
                    container.style.left = `${rect.left + window.scrollX}px`;
                    container.style.top = `${rect.top + window.scrollY}px`;
                }
            };

            this.addCustomWidget(widget);

            const originalOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                const containerId = `qwen-prompt-container-${this.id}`;
                const container = document.getElementById(containerId);
                if (container) {
                    container.remove();
                }
                originalOnRemoved?.apply(this, arguments);
            };

            const originalOnExecuted = this.onExecuted;
            this.onExecuted = function (message) {
                originalOnExecuted?.apply(this, arguments);
                
                const promptData = message?.qwen_prompts;
                if (promptData) {
                    // 更新显示
                    const container = getOrCreateContainer(this);
                    const contentHeight = renderPromptsToContainer(container, promptData);
                    
                    // 限制最大高度 400px
                    const MAX_HEIGHT = 400;
                    const neededHeight = Math.min(contentHeight, MAX_HEIGHT) + 20;

                    // 存储prompt数据和高度
                    setNodePrompts(this, promptData, neededHeight);
                    
                    // 更新节点实例的widget高度
                    this.qwenWidgetHeight = neededHeight;

                    if (this.onResize) {
                        this.onResize(this.size);
                    }
                    this.graph?.setDirtyCanvas(true, true);
                }
            };
        };
    },
});
