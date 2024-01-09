from setuptools import setup, find_packages

with open("./README.md", "r") as fh:
    long_description = fh.read()
setup(
    name="llama_memory",
    version="0.0.1a0",
    packages=find_packages(),
    install_requires=[
        "util_helper>=0.0.4",
        "chromadb==0.4.22"
        "gguf_llama>=0.1.2"
    ],
    include_package_data=True,
    author="Łael Al-Halawani",
    author_email="laelhalawani@gmail.com",
    description="Easy deployment of quantized llama models on cpu",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3.1",
        "Topic :: Text Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['vector database', 'vector db', 'rag', 'long term ai memory', 'llama', 'ai', 'artificial intelligence', 'natural language processing', 'nlp', 'quantization', 'cpu', 'deployment', 'inference', 'model', 'models', 'model database', 'model repo', 'model repository', 'model library', 'model libraries',
              'gguf', 'llm cpu', 'llm'],
    url="https://github.com/laelhalawani/llama_memory",
)