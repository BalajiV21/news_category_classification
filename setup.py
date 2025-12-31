"""
Setup script for News Category Classification package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="news-category-classifier",
    version="1.0.0",
    author="Balaji Viswanathan",
    author_email="balaji@example.com",
    description="A bidirectional LSTM-based classifier for news category classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BalajiV21/news_category_classification",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "news-train=news_classifier.train:main",
            "news-evaluate=news_classifier.evaluate:main",
            "news-predict=news_classifier.predict:main",
        ],
    },
)
