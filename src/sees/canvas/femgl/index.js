const regl = require('regl')({
  extensions: 'OES_element_index_uint'
})
const camera = require('./camera')({regl})
const createMesh = require('./fem')({regl})

const state = {
  center: [0, 0, 0],
  eye: [0, 0, 0],
  up: [0, 1, 0],
  polar: [Math.PI / 4, Math.PI / 16, 0],
  dpolar: [0, 0, 0],
  displacement: 0,
  lineWidth: 1.25,
  mode: 'stress',
  elements: true,
  lines: true,
  ortho: true,
  subdivisions: 2,
  meshData: {} // require('./mesh.json') // fetch("./mesh.json").json() //
}

let mesh = null

async function rebuildMesh () {
  const response = await fetch("./mesh.json");
  state.meshData = await response.json();
  mesh = createMesh(state.meshData, state.subdivisions)
//mesh = createMesh(response.json(), state.subdivisions)
  state.center = mesh.center.slice()
  state.polar[0] = Math.PI / 4
  state.polar[1] = Math.PI / 16
  state.polar[2] = Math.log(2 * mesh.radius)
}

rebuildMesh();

function handleFiles ([file]) {
  const reader = new window.FileReader()
  reader.onload = (data) => {
    /* try { */
      const meshData = JSON.parse(data.target.result)
      if (!meshData["displacements"])
        meshData["displacements"] = new Array(meshData["coordinates"].length).fill([0, 0, 0]);
  
      mesh = createMesh(meshData, state.subdivisions)
      state.meshData = meshData
      rebuildMesh()
    /*
    } catch (e) {
      window.alert('invalid data file')
    }
    */
  }
  reader.readAsText(file)
}

const uploadInput = document.createElement('input')
uploadInput.setAttribute('type', 'file')
uploadInput.addEventListener('change', () => {
  if (uploadInput.files && uploadInput.files.length > 0) {
    handleFiles(uploadInput.files)
  }
})

require('control-panel')([
  {
    type: 'range',
    label: 'displacement',
    min: 0,
    max: 255,
    initial: state.displacement
  },
  /*
  {
    type: 'range',
    label: 'lineWidth',
    min: 0,
    max: 10,
    initial: state.lineWidth
  },
  */
  {
    type: 'select',
    label: 'mode',
    options: [
      'stress',
      'x',
      'y',
      'z',
      'total'
    ],
    initial: state.mode
  },
  {
    type: 'checkbox',
    label: 'ortho',
    initial: state.ortho
  },
  {
    type: 'checkbox',
    label: 'elements',
    initial: state.elements
  },
  {
    type: 'checkbox',
    label: 'lines',
    initial: state.lines
  },
  {
    type: 'range',
    label: 'subdivisions',
    min: 1,
    max: 8,
    step: 1,
    initial: state.subdivisions
  },
  {
    type: 'button',
    label: 'open file',
    action: () => {
      uploadInput.click()
    }
  }
]).on('input', (data) => {
  const psubdiv = state.subdivisions
  Object.assign(state, data)
  if (psubdiv !== data.subdivisions) {
    rebuildMesh()
  }
})

require('./gesture')({
  canvas: regl._gl.canvas,

  onZoom (dz) {
    state.dpolar[2] += 0.25 * dz
  },

  onRotate (dx, dy) {
    state.dpolar[0] += dx
    state.dpolar[1] -= dy
  }
})

require('drag-and-drop-files')(regl._gl.canvas, handleFiles)

regl.frame(({tick}) => {
  camera.integrate(state)

  regl.clear({
    color: [0, 0, 0, 0],
    depth: 1
  })

  camera.setup(state, () => {
    mesh.draw(state)
  })
})
