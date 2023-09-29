# `sees`

<!--
<img align="left" src="https://raw.githubusercontent.com/BRACE2/OpenSeesRT/master/docs/figures/banner.png" width="150px" alt="OpenSees">
-->

<img align="left" src="https://github.com/BRACE2/OpenSeesRT/blob/master/docs/figures/hockling.png?raw=true" width="350px" alt="SEES Logo">

<!--
https://github.com/claudioperez/sees/blob/master/docs/figures/hockling.png?raw=true
-->


**A modern OpenSees renderer**


<br>

<!-- 
-------------------------------------------------------------------- 
-->

<div style="align:center">

[![Latest PyPI version](https://img.shields.io/pypi/v/sees?logo=pypi&style=for-the-badge)](https://pypi.python.org/pypi/sees)
[![PyPI Downloads](https://img.shields.io/pypi/dm/sees?style=for-the-badge)](https://pypi.org/project/sees)

<!--
[![Latest conda-forge version](https://img.shields.io/conda/vn/conda-forge/sees?logo=conda-forge&style=for-the-badge)](https://anaconda.org/conda-forge/sees)
[![](https://img.shields.io/conda/v/sees/sees?color=%23660505&style=for-the-badge)](https://anaconda.org/sees/sees)
-->

</div>

<!-- 
-------------------------------------------------------------------- 
-->

`sees` is a finite element rendering library that leverages modern 
web technologies to produce sharable, efficient, and beautiful renderings.


<!-- Badge links -->

[pypi-d-image]: https://img.shields.io/pypi/dm/sees.svg
[license-badge]: https://img.shields.io/pypi/l/sees.svg
[pypi-d-link]: https://pypi.org/project/sees
[pypi-v-image]: https://img.shields.io/pypi/v/sees.svg
[pypi-v-link]: https://pypi.org/project/sees


-------------------------------------------------------------------- 

<br>

Documentation is currently under development.

## Features

- Extruded deformed shape

- Detailed section rendering

- A wide selection of rendering backends and output file types, including 
  optimized 3D web formats like `.glb`.

- Correctly render models that treat both `y` or `z` as the
  vertical coordinate. Just pass the  option `vert=3` to render
  model `z` vertically, or `vert=2` to render model `y` vertically.

-------------------------------------------------------------------- 

## Command Line Interface

To create a rendering, execute the following command from the anaconda prompt (after activating the appropriate environment):

```shell
python -m sees model.json -o model.html
```

If you omit the `-o <file.html>` portion, it will plot immediately in a new window. You can also use a `.png` extension to save a static image file, as opposed to the interactive html.


To plot an elevation (`elev`) plan (`plan`) or section (`sect`) view, run:

```shell
python -m sees model.json --view elev
```

and add `-o <file.extension>` as appropriate.

To see the help page run

```shell
python -m sees --help
```

<br>

See also

- [`opensees`](https://pypi.org/project/opensees)

