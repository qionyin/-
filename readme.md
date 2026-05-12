### 一、环境准备

#### 1. Python环境安装
- 安装 **Python 3.12** 或更高版本
- 确保pip已正确配置

#### 2. 克隆/复制项目文件
将整个 `PythonCST3.12` 文件夹复制到新电脑，例如：

D:\pythoncode\PythonCST3.12



---

### 二、依赖安装

#### 1. 创建虚拟环境（推荐）
powershell
cd D:\pythoncode\PythonCST3.12
python -m venv venv
.\venv\Scripts\Activate.ps1



#### 2. 安装Python依赖包
powershell
pip install dashscope chromadb python-dotenv



**主要依赖说明：**
- `dashscope`: 通义千问API客户端
- `chromadb`: 向量数据库（长期记忆）
- `python-dotenv`: 环境变量管理

---

### 三、配置设置

#### 1. 创建 `.env` 文件
在项目根目录创建 `.env` 文件：

env
# 通义千问API密钥（必需）
DASHSCOPE_API_KEY=sk-your_api_key_here

# CST项目默认路径（可选）
CST_PROJECT_PATH=D:\Projects\antenna.cst

# CST结果默认路径（可选）
CST_RESULTS_PATH=D:\Results\simulation_001



#### 2. 获取通义千问API密钥
- 访问阿里云DashScope官网注册
- 创建API Key并填入 `.env` 文件
#如只需简单测试，在此就可以直接运行测试问答rag，步骤四是测试cst自动仿真rag所需
---

### 四、CST软件配置（如使用CST优化功能）

#### 1. 确保CST Studio Suite已安装
- 需要在新电脑上安装CST软件
- 确认CST COM接口可用

#### 2. 准备CST项目文件
- 将 `.cst` 项目文件复制到新电脑
- 更新 `.env` 中的路径配置

---

### 五、首次运行测试

#### 1. 启动系统
```powershell
python main.py
```


#### 2. 选择运行模式
系统会提示选择：
- **选项1**: 通用RAG问答系统（纯对话）
- **选项2**: CST仿真自动优化系统

#### 3. 验证功能
- 测试对话功能是否正常
- 检查ChromaDB向量库是否自动创建
- 确认LLM调用成功



### 六、运行模式选择

#### 快速模式（推荐初次使用）
python
# 在 main.py 中已默认启用
use_fast_mode=True


- ✅ 无需生成向量，启动快
- ✅ 适合测试和演示

#### 向量模式（生产环境）
python
use_fast_mode=False


- ✅ 支持语义搜索，更准确
- ⚠️ 首次运行需生成嵌入向量

---






