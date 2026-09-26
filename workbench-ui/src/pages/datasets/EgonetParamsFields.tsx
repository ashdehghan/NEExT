export type EgonetMethod = "k_hop" | "random_walk";
export type EgonetNodeSelection = "all_nodes" | "sample_fraction" | "specific_node_ids";

export interface EgonetParamsState {
  egonetMethod: EgonetMethod;
  kHop: number;
  walkLength: number;
  nWalks: number;
  restartProb: number;
  nodeSelection: EgonetNodeSelection;
  sampleFraction: number;
  randomSeed: number;
  sourceNodeIdsText: string;
  targetNodeAttribute: string;
}

export const DEFAULT_EGONET_PARAMS: EgonetParamsState = {
  egonetMethod: "k_hop",
  kHop: 1,
  walkLength: 10,
  nWalks: 100,
  restartProb: 0.15,
  nodeSelection: "all_nodes",
  sampleFraction: 1,
  randomSeed: 13,
  sourceNodeIdsText: "",
  targetNodeAttribute: ""
};

export function egonetParamsPayload(state: EgonetParamsState, filterLargestComponent: boolean, sourceNodeIds: string[]) {
  return {
    graph_type: "networkx" as const,
    filter_largest_component: filterLargestComponent,
    egonet_method: state.egonetMethod,
    k_hop: state.kHop,
    walk_length: state.walkLength,
    n_walks: state.nWalks,
    restart_prob: state.restartProb,
    node_selection: state.nodeSelection,
    sample_fraction: state.nodeSelection === "sample_fraction" ? state.sampleFraction : 1,
    random_seed: state.randomSeed,
    source_node_ids: state.nodeSelection === "specific_node_ids" ? sourceNodeIds : [],
    target_node_attribute: state.targetNodeAttribute || null
  };
}

interface EgonetParamsFieldsProps {
  state: EgonetParamsState;
  onChange: (patch: Partial<EgonetParamsState>) => void;
  attributeOptions: string[];
}

/**
 * The single-graph egonet parameter form, shared by the import and
 * configure-from-library views. The random seed is shown whenever it
 * matters: for sampled centers and always for random walks.
 */
export function EgonetParamsFields({ state, onChange, attributeOptions }: EgonetParamsFieldsProps) {
  const showSeed = state.nodeSelection === "sample_fraction" || state.egonetMethod === "random_walk";
  return (
    <>
      <label className="field">
        <span>Egonet Method</span>
        <select value={state.egonetMethod} onChange={(event) => onChange({ egonetMethod: event.target.value as EgonetMethod })}>
          <option value="k_hop">K-hop neighborhood</option>
          <option value="random_walk">Random walk</option>
        </select>
      </label>
      {state.egonetMethod === "k_hop" ? (
        <label className="field">
          <span>K-Hop</span>
          <input type="number" min={0} max={10} value={state.kHop} onChange={(event) => onChange({ kHop: Number(event.target.value) })} />
        </label>
      ) : (
        <>
          <label className="field">
            <span>Walk Length</span>
            <input
              type="number"
              min={1}
              max={1000}
              value={state.walkLength}
              onChange={(event) => onChange({ walkLength: Number(event.target.value) })}
            />
          </label>
          <label className="field">
            <span>Walks per Node</span>
            <input
              type="number"
              min={1}
              max={100000}
              value={state.nWalks}
              onChange={(event) => onChange({ nWalks: Number(event.target.value) })}
            />
          </label>
          <label className="field">
            <span>Restart Probability</span>
            <input
              type="number"
              min={0}
              max={0.99}
              step={0.01}
              value={state.restartProb}
              onChange={(event) => onChange({ restartProb: Number(event.target.value) })}
            />
          </label>
        </>
      )}
      <label className="field">
        <span>Node Selection</span>
        <select
          value={state.nodeSelection}
          onChange={(event) => onChange({ nodeSelection: event.target.value as EgonetNodeSelection })}
        >
          <option value="all_nodes">All nodes</option>
          <option value="sample_fraction">Sample fraction</option>
          <option value="specific_node_ids">Specific node IDs</option>
        </select>
      </label>
      <label className="field">
        <span>Target Attribute</span>
        <select value={state.targetNodeAttribute} onChange={(event) => onChange({ targetNodeAttribute: event.target.value })}>
          <option value="">None</option>
          {attributeOptions.map((column) => (
            <option key={column} value={column}>
              {column}
            </option>
          ))}
        </select>
      </label>
      {state.nodeSelection === "sample_fraction" ? (
        <label className="field">
          <span>Sample Fraction</span>
          <input
            type="number"
            min={0.01}
            max={1}
            step={0.01}
            value={state.sampleFraction}
            onChange={(event) => onChange({ sampleFraction: Number(event.target.value) })}
          />
        </label>
      ) : null}
      {showSeed ? (
        <label className="field">
          <span>Random Seed</span>
          <input type="number" min={0} value={state.randomSeed} onChange={(event) => onChange({ randomSeed: Number(event.target.value) })} />
        </label>
      ) : null}
      {state.nodeSelection === "specific_node_ids" ? (
        <label className="field field-wide">
          <span>Source Node IDs</span>
          <textarea
            value={state.sourceNodeIdsText}
            rows={4}
            onChange={(event) => onChange({ sourceNodeIdsText: event.target.value })}
          />
        </label>
      ) : null}
    </>
  );
}
