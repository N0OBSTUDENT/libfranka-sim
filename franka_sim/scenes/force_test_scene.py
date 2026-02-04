import genesis as gs
from genesis.options.sensors import ContactForce

def init_scene(scene: gs.Scene):

    # contact_origin = gs.morphs.Box(pos=(0.0, 0.0, 0.025),
    #                             size=(0.05, 0.05, 0.05),
    #                             fixed=True)
    # origin_entity = scene.add_entity(contact_origin)


    

    contact_base = gs.morphs.Box(pos=(0.2, 0.0, 0.0175),
                                size=(0.8, 0.8, 0.015),
                                fixed=True)
    base_entity = scene.add_entity(contact_base)

    sensor = scene.add_sensor(
        ContactForce(
            entity_idx = base_entity.idx,
            link_idx_local = 0,
            draw_debug=True,
        )
    )
    plot_kwargs = dict(
                title=f"Base Force Sensor Data",
                labels=["force_x", "force_y", "force_z"],
            )
    sensor.start_recording(gs.recorders.MPLLinePlot(**plot_kwargs))

    # contact_box = gs.morphs.Box(pos=(0.7, 0.0, 0.5),
    #                             size=(0.1, 0.8, 0.8),
    #                             fixed=True)
    # box_entity = scene.add_entity(contact_box)

    # sensor = scene.add_sensor(
    #     ContactForce(
    #         entity_idx = box_entity.idx,
    #         link_idx_local = 0,
    #         draw_debug=True,
    #     )
    # )
    # plot_kwargs = dict(
    #             title=f"Box Force Sensor Data",
    #             labels=["force_x", "force_y", "force_z"],
    #         )
    # sensor.start_recording(gs.recorders.MPLLinePlot(**plot_kwargs))
    pass