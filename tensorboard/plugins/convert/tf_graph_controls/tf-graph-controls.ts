/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
namespace tf.graph.controls {

interface DeviceNameExclude {
  regex: RegExp,
}

const DEVICE_NAME_REGEX = /device:([^:]+:[0-9]+)$/;

/**
 * Display only devices matching one of the following regex.
 */
const DEVICE_NAMES_INCLUDE: DeviceNameExclude[] = [
  {
    // Don't include GPU stream, memcpy, etc. devices
    regex: DEVICE_NAME_REGEX,
  }
];

interface StatsDefaultOff {
  regex: RegExp,
  msg: string, // 'Excluded by default since...'
}

/**
 * Stats from device names that match these regexes will be disabled by default.
 * The user can still turn on a device by selecting the checkbox in the device list.
 */
const DEVICE_STATS_DEFAULT_OFF: StatsDefaultOff[] = [];

export interface Selection {
  run: string,
  tag: string | null,
  type: tf.graph.SelectionType,
}

export interface DeviceForStats {
  [key: string]: boolean;
}

// TODO(stephanwlee): Move this to tf-graph-dashboard
export interface TagItem {
  tag: string | null,
  displayName: string,
  conceptualGraph: boolean,
  opGraph: boolean,
  profile: boolean,
}

// TODO(stephanwlee): Move this to tf-graph-dashboard
export interface RunItem {
  name: string,
  tags: TagItem[],
}

// TODO(stephanwlee): Move this to tf-graph-dashboard
export type Dataset = Array<RunItem>;

interface CurrentDevice {
  device: string,
  suffix: string,
  used: boolean,
  ignoredMsg: string | null,
}

export enum ColorBy {
  COMPUTE_TIME = 'compute_time',
  MEMORY = 'memory',
  STRUCTURE = 'structure',
  XLA_CLUSTER = 'xla_cluster',
  OP_COMPATIBILITY = 'op_compatibility',
}

interface ColorParams {
  minValue: number,
  maxValue: number,
  // HEX value describing color.
  startColor: string,
  // HEX value describing color.
  endColor: string,
}

interface DeviceColor {
  device: string;
  color: string;
}

interface XlaClusterColor {
  xla_cluster: string;
  color: string;
}

// TODO(stephanwlee) Move this to tf-graph.html when it becomes TypeScript.
interface ColorByParams {
  compute_time: ColorParams,
  memory: ColorParams,
  device: DeviceColor[],
  xla_cluster: XlaClusterColor[],
}

const GRADIENT_COMPATIBLE_COLOR_BY: Set<ColorBy> = new Set([
    ColorBy.COMPUTE_TIME, ColorBy.MEMORY]);


Polymer({
  is: 'tf-graph-controls',
  properties: {
    // TODO:
    selectedGraph:String,
    graph: Object,
    graphHierarchy: {
      type: Object,
      notify: true,
    },
    param:{
      type: Object,
    },
    srcPath: String,
    desPath: String,
    srcType: String,
    desType: String,
    // Public API.
    /**
     * @type {?tf.graph.proto.StepStats}
     */
    stats: {
      value: null,
      type: Object,
      // observer: '_statsChanged',
    },
    
    /**
     * @type {!tf.graph.controls.ColorBy}
     */
    colorBy: {
      type: String,
      value: ColorBy.STRUCTURE,
      notify: true,
    },
    colorByParams: {
      type: Object,
      notify: true,
      // TODO(stephanwlee): Change readonly -> readOnly and fix the setter.
      readonly: true,
    },
    datasets: {
      type: Array,
      observer: '_datasetsChanged',
      value: () => [],
    },
    /**
     * @type {tf.graph.render.RenderGraphInfo}
     */
    renderHierarchy: {
      type: Object,
      notify: true,
    },
    /**
     * @type {!Selection}
     */
    selection: {
      type: Object,
      notify: true,
      readOnly: true,
      computed: '_computeSelection(datasets, _selectedRunIndex, _selectedTagIndex, _selectedGraphType)',
    },
    
    _selectedRunIndex: {
      type: Number,
      value: 0,
      observer: '_selectedRunIndexChanged',
    },
    _selectedTagIndex: {
      type: Number,
      value: 0,
      observer: '_selectedTagIndexChanged',
    },
    /**
     * @type {tf.graph.SelectionType}
     */
    _selectedGraphType: {
      type: String,
      value: tf.graph.SelectionType.OP_GRAPH,
    },
    selectedNode: {
      type: String,
      notify: true,
    },
    selectedNodeInclude: {
      type: Number,
      notify: true,
    },
    highlightedNode: {
      type: String,
      notify: true,
    },
    left_width:{
      type: Number,
      value: 40,
    },
    right_width:{
      type: Number,
      value: 40,
    }
  },

  listeners: {},

  ready: function(){},
  dragLeft(){
    var left = document.getElementsByClassName('source-graphboard') as HTMLCollectionOf<HTMLElement>;
    var right = document.getElementsByClassName('target-graphboard') as HTMLCollectionOf<HTMLElement>;
    // left[0].style.width = '20%'
    this.left_width -= 2
    this.right_width += 2 
    left[0].style.width = this.left_width.toString() + '%'
    right[0].style.width = this.right_width.toString() + '%'
    // console.info(left[0],right[0])
    // console.info('left')
  },
  dragRight(){
    var left = document.getElementsByClassName('source-graphboard') as HTMLCollectionOf<HTMLElement>;
    var right = document.getElementsByClassName('target-graphboard') as HTMLCollectionOf<HTMLElement>;
    // left[0].style.width = '20%'
    this.left_width += 2
    this.right_width -= 2 
    left[0].style.width = this.left_width.toString() + '%'
    right[0].style.width = this.right_width.toString() + '%'
    // console.info('right')
  },
  _isGradientColoring: function(
      stats: tf.graph.proto.StepStats, colorBy: ColorBy): boolean {
    return GRADIENT_COMPATIBLE_COLOR_BY.has(colorBy) && stats != null;
  },

  _datasetsChanged: function(newDatasets: Dataset, oldDatasets: Dataset) {
    if (oldDatasets != null) {
      // Select the first dataset by default.
      this._selectedRunIndex = 0;
    }
  },

  _computeSelection: function(datasets: Dataset, _selectedRunIndex: number,
      _selectedTagIndex: number, _selectedGraphType: tf.graph.SelectionType) {
    if (!datasets[_selectedRunIndex] ||
        !datasets[_selectedRunIndex].tags[_selectedTagIndex]) {
      return null;
    }

    return {
      run: datasets[_selectedRunIndex].name,
      tag: datasets[_selectedRunIndex].tags[_selectedTagIndex].tag,
      type: _selectedGraphType,
    }
  },

  _selectedRunIndexChanged: function(runIndex: number): void {
    if (!this.datasets) return;
    // Reset the states when user pick a different run.
    this.colorBy = ColorBy.STRUCTURE;
    this._selectedTagIndex = 0;
    this._selectedGraphType = this._getDefaultSelectionType();
  },

  _selectedTagIndexChanged(): void {
      this._selectedGraphType = this._getDefaultSelectionType();
  },

  _getDefaultSelectionType(): tf.graph.SelectionType {
    const {
      datasets,
      _selectedRunIndex: run,
      _selectedTagIndex: tag,
    } = this;
    if (!datasets ||
        !datasets[run] ||
        !datasets[run].tags[tag] ||
        datasets[run].tags[tag].opGraph) {
      return tf.graph.SelectionType.OP_GRAPH;
    }
    if (datasets[run].tags[tag].profile) {
      return tf.graph.SelectionType.PROFILE;
    }
    if (datasets[run].tags[tag].conceptualGraph) {
      return tf.graph.SelectionType.CONCEPTUAL_GRAPH;
    }
    return tf.graph.SelectionType.OP_GRAPH;
  },

  showSuccess: function(){
    console.info('success')
  },

  transformModel: function(){
    console.info(this.srcPath,this.desPath,this.srcType,this.desType)
  },
});

}  // namespace tf.graph.controls
