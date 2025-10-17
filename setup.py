from setuptools import setup, find_packages

setup(
    name="medimaging-ai",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "streamlit",
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "pyyaml",
        "plotly",
        "pandas",
    ],
    python_requires=">=3.8",
)