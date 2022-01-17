use std::f32::consts::{FRAC_PI_4};
use bevy::{core::FixedTimestep, prelude::*};
use chain::{chain_mesh_system, chain_system, setup_chains, Chain};
use cyclic_coordinate_descent::cyclic_coordinate_descent;
use jacobian_pseudo_inverse::jacobian_pseudo_inverse;
use jacobian_transpose::jacobian_transpose;

mod chain;
mod jacobian_pseudo_inverse;
mod jacobian_transpose;
mod cyclic_coordinate_descent;


fn setup(
	mut commands: Commands,
	mut meshes: ResMut<Assets<Mesh>>,
	mut materials: ResMut<Assets<StandardMaterial>>,
) {
	commands.spawn_bundle(PbrBundle {
		mesh: meshes.add(Mesh::from(shape::Plane { size: 40.0 })),
		material: materials.add(Color::rgb(0.3, 0.5, 0.3).into()),
		..Default::default()
	});

	commands.spawn_bundle(DirectionalLightBundle {
		directional_light: DirectionalLight {
			illuminance: 60000.0,
			shadows_enabled: true,
			..Default::default()
		},
		transform: Transform {
			translation: Vec3::new(0.0, 2.0, 0.0),
			rotation:
				Quat::from_rotation_y(FRAC_PI_4) *
				Quat::from_rotation_x(-FRAC_PI_4),
			..Default::default()
		},

		..Default::default()
	});

	commands.insert_resource(AmbientLight {
		color: Color::WHITE,
		brightness: 0.3,
	});

	commands.spawn_bundle(PerspectiveCameraBundle {
		transform: Transform::from_xyz(-25.0, 20.0, 25.0).looking_at(Vec3::ZERO, Vec3::Y),
		..Default::default()
	});
}

fn main() {
	App::new()
		.insert_resource(WindowDescriptor {
			title: "IK Test".to_string(),
			width: 1280.0,
			height: 720.0,
			vsync: false,
			position: Some(Vec2::new(0.0, 0.0)),
			..Default::default()
		})
		.add_plugins(DefaultPlugins)
		.insert_resource(ClearColor(Color::rgb(0.125, 0.25, 0.5)))
		.add_startup_system(setup)
		.add_startup_system(setup_chains)
		.add_system_set(SystemSet::new()
			.with_run_criteria(FixedTimestep::step(1.0/60.0))
			.with_system(chain_system))
		.add_system_set(SystemSet::new()
			.with_run_criteria(FixedTimestep::step(1.0/120.0))
			.with_system(chain_mesh_system))
		.run();
}
