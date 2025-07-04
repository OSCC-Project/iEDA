import ipower_cpp

work_dir = "/home/taosimin/nangate45/design/example"

ipower_cpp.set_design_workspace(work_dir + "/rpt")
ipower_cpp.read_netlist(work_dir + "/example1.v")
ipower_cpp.read_liberty(
    ["/home/taosimin/nangate45/lib/NangateOpenCellLibrary_typical.lib"]
)
ipower_cpp.link_design("top")
ipower_cpp.read_sdc(work_dir + "/example1.sdc")

ipower_cpp.create_data_flow()
connection_map = ipower_cpp.build_connection_map(
    [{"r1", "u1"}, {"r2", "u2"}, {"r3"}, {"in1"}, {"in2"}, {"out"}], {"r1", "u1"}, 2
)

for src_cluster_id, snk_clusters in connection_map.items():
    for snk_cluster in snk_clusters:
        print(
            "src cluster id {} -> snk cluster id {} stages {} hop {}".format(
                src_cluster_id,
                snk_cluster.dst_cluster_id,
                " ".join([str(x) for x in snk_cluster.stages_each_hop]),
                snk_cluster.hop,
            )
        )


macro_connections = ipower_cpp.build_macro_connection_map(2)
print("build macro connections finished")
for macro_connection in macro_connections:
    print(
        "src macro name {} -> snk macro name {} stages {} hop {}".format(
            macro_connection.src_macro_name,
            macro_connection.dst_macro_name,
            " ".join([str(x) for x in snk_cluster.stages_each_hop]),
            macro_connection.hop,
        )
    )
