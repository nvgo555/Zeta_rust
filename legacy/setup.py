from setuptools import setup, find_packages

setup(
    name="zeta-p-adic-ai-unofficial",
    version="7.0.0-experimental",
    author="Dávid Navrátil (Original), Unofficial Fork Maintainer",
    author_email="david.navratil2016@gmail.com",
    description="[EXPERIMENTAL UNOFFICIAL] Algebraic AI built exclusively on Z_13[eta] — zero float, zero gradient, zero Euclidean geometry",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="CC-BY-NC-4.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
)
