#!/usr/bin/python

from numpy.distutils.core import setup, Extension

ext_hmmf = Extension(name="pyvb._hmmf",sources=["pyvb/_hmmf.f90",])
ext_hmmc = Extension(name="pyvb._hmmc",sources=["pyvb/_hmmc.c",])

setup(name='pyvb',
        version='1.2',
        description='Python implementation for Variational Gaussiam Mixiture Model',
        author='Shiqiao Du',
        author_email='lucidfrontier.45@gmail.com',
        url='http://frontier45.web.fc2.com/',
        packages=['pyvb'],
        ext_modules = [ext_hmmf,ext_hmmc]
        )
