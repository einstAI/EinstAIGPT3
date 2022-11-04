
//
// Energy models need to be dynamic
//




if (relevantScope[{node_scope}]) {{

//
// Energy models need to be dynamic
//



{node_energy}

}} // end if relevantScope



    // notNanPerNode[{node_id}] = true;
    {floating_data_type} probsNode{node_id}[] = {{ {node_p} }};
    {floating_data_type} probsNode{node_id}_sum = 0;
    for (int i = 0; i < {node_num_states}; i++) {{
        probsNode{node_id}_sum += probsNode{node_id}[i];
    }}

    //not null condition
    if (nullValueIdx{node_scope} != -1) {{
        nodeIntermediateResult[{node_id}] = 1 - probsNode{node_id}[nullValueIdx{node_scope}];
    }} else {{
        for (int &idx: possibleValues{node_scope}) {{
            nodeIntermediateResult[{node_id}] += probsNode{node_id}[idx];
        }}
    }}
    {final_assert}

}}