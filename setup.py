from setuptools import setup, find_packages

setup(
    name="zeta-p-adic-ai",
    version="7.0.0",
    author="Dávid Navrátil",
    author_email="david.navratil2016@gmail.com",
    description="Algebraic AI built exclusively on Z_13[eta] — zero float, zero gradient, zero Euclidean geometry",
    license="CC-BY-NC-4.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
)
