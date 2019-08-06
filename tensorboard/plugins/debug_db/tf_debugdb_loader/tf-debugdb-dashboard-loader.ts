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
namespace tf.graph.loader {

interface GraphRunTag {
  run: string;
  tag?: string;
}

Polymer({
  is: 'tf-debugdb-dashboard-loader',

  properties: {
    datasets: Array,
    /**
     * @type {{value: number, msg: string}}
     *
     * A number between 0 and 100 denoting the % of progress
     * for the progress bar and the displayed message.
     */
    progress: {
      type: Object,
      notify: true,
    },
    selection: Object,
    compatibilityProvider: {
      type: Object,
      value: () => new tf.graph.op.TpuCompatibilityProvider(),
    },
    hierarchyParams: {
      type: Object,
      value: () => tf.graph.hierarchy.DefaultHierarchyParams,
    },
    outGraphHierarchy: {
      type: Object,
      notify: true
    },
    outGraph: {
      type: Object,
      notify: true
    },
    /** @type {Object} */
    outStats: {
      type: Object,
      notify: true
    },
    /**
     * @type {?GraphRunTag}
     */
    _graphRunTag: Object,
  },
  observers: [],
  _load: function(loadparams): Promise<void> {
    // Clear stats about the previous graph.
    this._setOutStats(null);
    const params = new URLSearchParams();
    params.set('source_type', loadparams.modelType);
    if(loadparams.modelType == 'caffe2'){
      params.set('file_type', loadparams.fileType);
      params.set('predict_net',loadparams.predictNet)
      params.set('init_net',loadparams.initNet)
    }
    else{
      if(loadparams.modelType == 'torch'){
        params.set('input_tensor_size', loadparams.inputTensorSize);
        params.set('model_file', loadparams.srcPath);
      }
      else{
        params.set('file_type', loadparams.fileType);
        params.set('model_file', loadparams.srcPath);
      }
    }
    
    const graphPath =
        tf_backend.getRouter().pluginRoute('graphedit', '/load', params);
    return this._fetchAndConstructHierarchicalGraph(graphPath).then(() => {
      // console.info('yes')
    })
    return ;
      
  },

  _fetchAndConstructHierarchicalGraph: async function(
    path: (string|null), pbTxtFile?: Blob): Promise<void> {
  // Reset the progress bar to 0.
  this.set('progress', {
    value: 0,
    msg: '',
  });
  const tracker = tf.graph.util.getTracker(this);
  return tf.graph.loader.fetchAndConstructHierarchicalGraph(
    tracker,
    path,
    pbTxtFile,
    this.compatibilityProvider,
    this.hierarchyParams,
  ).then(({graph, graphHierarchy}) => {
    this.outGraph = graph
    this.outGraphHierarchy = graphHierarchy
  });
},
});

}  // namespace tf.graph.loader
