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
namespace tf.convert.controls {

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
  type: tf.convert.SelectionType,
}

export interface DeviceForStats {
  [key: string]: boolean;
}

// TODO(stephanwlee): Move this to tf-convert-dashboard
export interface TagItem {
  tag: string | null,
  displayName: string,
  conceptualGraph: boolean,
  opGraph: boolean,
  profile: boolean,
}

// TODO(stephanwlee): Move this to tf-convert-dashboard
export interface RunItem {
  name: string,
  tags: TagItem[],
}

// TODO(stephanwlee): Move this to tf-convert-dashboard
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

// TODO(stephanwlee) Move this to tf-convert.html when it becomes TypeScript.
interface ColorByParams {
  compute_time: ColorParams,
  memory: ColorParams,
  device: DeviceColor[],
  xla_cluster: XlaClusterColor[],
}

const GRADIENT_COMPATIBLE_COLOR_BY: Set<ColorBy> = new Set([
    ColorBy.COMPUTE_TIME, ColorBy.MEMORY]);


Polymer({
  is: 'tf-convert-controls',
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
    srcType: Number,
    desType: Number,
    _modelTypes:{
      type: Array,
      value: ['onnx', 'caffe', 'caffe2', 'torch', 'tf'],
    },
    // Public API.
    /**
     * @type {?tf.convert.proto.StepStats}
     */
    stats: {
      value: null,
      type: Object,
      // observer: '_statsChanged',
    },
    
    /**
     * @type {!tf.convert.controls.ColorBy}
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
    /**
     * @type {!Selection}
     */
    selection: {
      type: Object,
      notify: true,
    },
    targetParams: {
      type: Object,
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

  transformModel: function(){
    var data = {
      'destination_path': this.desType,
      'destination_type': this._modelTypes[this.desType],
    }
    this.targetParams = data
    console.info(this.targetParams)
  },

  loadModel: function(){
    var data = {
      'source_path': this.srcPath,
      'source_type': this._modelTypes[this.srcType],
    }
    this.selection = data
  },

});

}  // namespace tf.convert.controls
