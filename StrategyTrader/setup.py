from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="strategy-trader",
    version="1.0.0",
    author="Strategy Trader Team",
    author_email="",
    description="Sistema completo de desarrollo, backtesting y optimización de estrategias de trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/strategy-trader",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "full": ["scikit-optimize>=0.9.0"],
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "flake8>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "strategy-trader=src.cli:main",
        ],
    },
)
