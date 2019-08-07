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
namespace tf.convert.loader {

interface GraphRunTag {
  run: string;
  tag?: string;
}

Polymer({
  is: 'tf-convert-dashboard-loader',

  properties: {
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
      observer:'_load'
    },
    targetParams: {
      type: Object,
      observer:'_tranform'
    },
    /**
     * TODO(stephanwlee): This should be changed to take in FileList or
     * the prop should be changed to `fileInput`.
     * @type {?Event}
     */
    compatibilityProvider: {
      type: Object,
      value: () => new tf.convert.op.TpuCompatibilityProvider(),
    },
    hierarchyParams: {
      type: Object,
      value: () => tf.convert.hierarchy.DefaultHierarchyParams,
    },
    outGraphHierarchy: {
      type: Object,
      notify: true
    },
    outGraph: {
      type: Object,
      notify: true
    },
    outGraphHierarchyTar: {
      type: Object,
      notify: true
    },
    outGraphTar: {
      type: Object,
      notify: true
    },
    /** @type {Object} */
    outStats: {
      type: Object,
      readOnly: true, // This property produces data.
      notify: true
    },
    /**
     * @type {?GraphRunTag}
     */
    _graphRunTag: Object,
  },
  observers: [],
  _load: function(): Promise<void> {
    const params = new URLSearchParams();
    params.set('source_path', this.selection.source_path);
    params.set('source_type', this.selection.source_type);
    
    const graphPath =
        tf_backend.getRouter().pluginRoute('convert', '/load', params);
    return this._fetchAndConstructHierarchicalGraph(graphPath).then(() => {
    })
    return ;
  },
  _tranform: function(): Promise<void> {
    const params = new URLSearchParams();
    params.set('destination_path', this.targetParams.destination_path);
    params.set('destination_type', this.targetParams.destination_type);
    
    const graphPath =
        tf_backend.getRouter().pluginRoute('convert', '/convert', params);
    return this._fetchAndConstructHierarchicalGraphTransform(graphPath).then(() => {
    })
    return ;
  },

  _fetchAndConstructHierarchicalGraph: async function(
      path: (string|null), pbTxtFile?: Blob): Promise<void> {
    // Reset the progress bar to 0.
    this.set('progress', {
      value: 100,
      msg: "Namespace hierarchy: Finding similar subgraphs",
    });
    const tracker = tf.convert.util.getTracker(this);
    return tf.convert.loader.fetchAndConstructHierarchicalGraph(
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

  _fetchAndConstructHierarchicalGraphTransform: async function(
    path: (string|null), pbTxtFile?: Blob): Promise<void> {
  // Reset the progress bar to 0.
  this.set('progress', {
    value: 100,
    msg: "Namespace hierarchy: Finding similar subgraphs",
  });
  const tracker = tf.convert.util.getTracker(this);
  return tf.convert.loader.fetchAndConstructHierarchicalGraph(
    tracker,
    path,
    pbTxtFile,
    this.compatibilityProvider,
    this.hierarchyParams,
  ).then(({graph, graphHierarchy}) => {
    this.outGraphTar = graph
    this.outGraphHierarchyTar = graphHierarchy
  });
},
});
}  // namespace tf.convert.loader
