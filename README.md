# `sees`

<!--
<img align="left" src="https://raw.githubusercontent.com/BRACE2/OpenSeesRT/master/docs/figures/banner.png" width="150px" alt="OpenSees">
-->

<img align="left" src="https://github.com/claudioperez/opensees/blob/master/docs/figures/hockling.png?raw=true" width="350px" alt="SEES Logo">

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
  vertical coordinate.

-------------------------------------------------------------------- 

## Command Line Interface

To create a rendering, execute the following command from the anaconda prompt (after activating the appropriate environment):

```shell
python -m sees model.json -o model.html
```

where `model.json` is a JSON file generate from executing the following OpenSees command:

```tcl
print -JSON model.json
```

If you omit the `-o <file.html>` portion, it will plot immediately in a new
window. You can also use a `.png` extension to save a static image file, as
opposed to the interactive html.

> **Note** Printing depends on the JSON output of a model. Several materials and
> elements in the OpenSeesPy and upstream OpenSees implementations do not
> correctly print to JSON. For the most reliable results, use the
> [`opensees`](https://pypi.org/project/opensees) package.

By default, the rendering treats the $y$ coordinate as vertical.
In order to manually control this behavior, just pass the  option 
`--vert 3` to render model $z$ vertically, or `--vert 2` to render model $y$ vertically.

If the [`opensees`](https://pypi.org/project/opensees) package is installed,
you can directly render a Tcl script without first printing to JSON, 
by just passing a Tcl script instead of the JSON file:

```shell
python -m sees model.tcl -o model.html
```

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

## Related Links

See also

- [`opensees`](https://github.com/claudioperez/opensees)
- [`osmg`](https://pypi.org/project/osmg)
- [`mdof`](https://pypi.org/project/mdof)
- [`sdof`](https://pypi.org/project/sdof)


The `sees` packages was used to generate figures for the following publications:

- *On nonlinear geometric transformations of finite elements* [doi: 10.1002/nme.7506](https://doi.org/10.1002/nme.7506)

Similar packages for OpenSees rendering include:

- [`vfo`](https://vfo.readthedocs.io/en/latest/)
- [`opsvis`](https://opsvis.readthedocs.io/en/latest/index.html)

Other

- [`fapp`](https://github.com/wcfrobert/fapp)

## Support

<table align="center">
<tr>

  <td>
    <a href="https://peer.berkeley.edu">
    <img src="https://raw.githubusercontent.com/claudioperez/sdof/master/docs/assets/peer-black-300.png"
         alt="PEER Logo" width="200"/>
    </a>
  </td>

  <td>
    <a href="https://dot.ca.gov/">
    <img src="https://raw.githubusercontent.com/claudioperez/sdof/master/docs/assets/Caltrans.svg.png"
         alt="Caltrans Logo" width="200"/>
    </a>
  </td>

  <td>
    <a href="https://brace2.herokuapp.com">
    <img src="https://raw.githubusercontent.com/claudioperez/sdof/master/docs/assets/stairlab.svg"
         alt="STAIRlab Logo" width="200"/>
    </a>
  </td>
 
 </tr>
</table>

