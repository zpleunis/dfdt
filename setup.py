import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dfdt",
    version="1.0.0",
    author="Ziggy Pleunis",
    author_email="ziggy.pleunis@physics.mcgill.ca",
    description="Linear drift rate measurements for fast radio bursts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zpleunis/dfdt",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
