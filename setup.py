from setuptools import setup

# setup(
#     name="desed_task",
#     version="0.1.1",
#     description="Sound Event Detection and Separation in Domestic Environments.",
#     author="DCASE Task 4 Organizers",
#     author_email="romain.serizel@loria.fr",
#     license="MIT",
#     packages=["desed_task"],
#     python_requires=">=3.8",
#     install_requires=[
#         "dcase_util>=0.2.16",
#         "psds_eval>=0.4.0",
#         "sed_eval>=0.2.1",
#         "sed_scores_eval>=0.0.0",
#     ],
# )



from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Ensure the output directory exists
os.makedirs("sed_scores_eval/base_modules", exist_ok=True)

extensions = [
    Extension(
        name="sed_scores_eval.base_modules.cy_detection",
        sources=["sed_scores_eval/base_modules/cy_detection.pyx"],
        include_dirs=[np.get_include()]
    ),
    Extension(
        name="sed_scores_eval.base_modules.cy_medfilt",
        sources=["sed_scores_eval/base_modules/cy_medfilt.pyx"],
        include_dirs=[np.get_include()],
    )

]

setup(
    name="desed_task",
    version="0.1.1",
    description="Sound Event Detection and Separation in Domestic Environments.",
    author="DCASE Task 4 Organizers",
    author_email="romain.serizel@loria.fr",
    license="MIT",
    packages=["desed_task"],
    python_requires=">=3.8",
    install_requires=[
        "dcase_util>=0.2.16",
        "psds_eval>=0.4.0",
        "sed_eval>=0.2.1",
        "sed_scores_eval>=0.0.0",
    ],
    #ext_modules=cythonize(extensions),
    ext_modules=cythonize(
        extensions,
        compiler_directives={'binding': True, 'linetrace': True}
    ),
    zip_safe=False,
)