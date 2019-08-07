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
namespace tf.debug.loader {

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
    selection: {
      type: Object,
      observer:'_newload'
    },
    attachParam: {
      type: Object,
      observer:'_attachload'
    },
    compatibilityProvider: {
      type: Object,
      value: () => new tf.debug.op.TpuCompatibilityProvider(),
    },
    hierarchyParams: {
      type: Object,
      value: () => tf.debug.hierarchy.DefaultHierarchyParams,
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
  _attachload: function(): Promise<void> {
    const params = new URLSearchParams();
    params.set('network_identification', this.attachParam);
    
    const graphPath =
        tf_backend.getRouter().pluginRoute('debugdb', '/attachload', params);
    return this._fetchAndConstructHierarchicalGraph(graphPath).then(() => {
      document.getElementById('graphboard').style.display = ''
    })
    return ;
  },

  _newload: function(): Promise<void> {
    var loadparams = this.selection
    const params = new URLSearchParams();
    params.set('model_type', loadparams.model_type);
    if(loadparams.modelType == 'caffe2'){
      params.set('file_type', loadparams.file_type);
      params.set('predict_net',loadparams.predict_net)
      params.set('init_net',loadparams.init_net)
    }
    else{
      if(loadparams.modelType == 'torch'){
        params.set('input_tensor_size', loadparams.input_tensor_size);
        params.set('source_path', loadparams.source_path);
      }
      else{
        params.set('file_type', loadparams.file_type);
        params.set('source_path', loadparams.source_path);
      }
    }
    
    const graphPath =
        tf_backend.getRouter().pluginRoute('debugdb', '/newload', params);
    return this._fetchAndConstructHierarchicalGraph(graphPath).then(() => {
      document.getElementById('graphboard').style.display = ''
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
  const tracker = tf.debug.util.getTracker(this);
  return tf.debug.loader.fetchAndConstructHierarchicalGraph(
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

}  // namespace tf.debug.loader
