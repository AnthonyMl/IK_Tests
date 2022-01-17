use std::f32::consts::PI;
use bevy::{prelude::*};
use itertools::{multizip, zip};
use rand::Rng;
use crate::{cyclic_coordinate_descent, jacobian_pseudo_inverse, jacobian_transpose};


#[allow(dead_code)]
pub enum Axis {
	X, Y, Z
}

impl Axis {
	pub fn to_vec3(&self) -> Vec3 {
		match *self {
			Axis::X => Vec3::new(1.0, 0.0, 0.0),
			Axis::Y => Vec3::new(0.0, 1.0, 0.0),
			Axis::Z => Vec3::new(0.0, 0.0, 1.0),
		}
	}
}

#[derive(PartialEq)]
pub enum ChainState {
	Seeking {
		target: Vec3,
		transition_frames: i32,
		current_frame: i32,
		start_angles: Vec<f32>,
		target_angles: Vec<f32>,
	},
	Waiting {
		wait_frames: i32,
	},
	Done,
}

impl Default for ChainState {
	fn default() -> Self { ChainState::Done }
}

pub struct Joint{
	pub axis: Axis,
	pub length: f32,
}

#[derive(Component)]
pub struct IsJoint;

#[derive(Component)]
pub struct IsTarget;

#[derive(Component)]
pub struct Chain {
	pub joints: Vec<Joint>,
	pub angles: Vec<f32>,
	pub state: ChainState,
	pub ik_fn: fn(&Chain, Vec3) -> Vec<f32>,
}

impl Chain {
	pub fn cumulative_transforms(&self) -> Vec<Mat4> {
		self.cumulative_transforms_with_angles(&self.angles)
	}

	pub fn cumulative_transforms_with_angles(&self, angles: &[f32]) -> Vec<Mat4> {
		let mut models = Vec::with_capacity(self.joints.len());
		let mut accumulator = Mat4::IDENTITY;

		for (joint, angle) in zip(&self.joints, angles) {
			let r = Mat4::from_axis_angle(joint.axis.to_vec3(), rad(*angle));
			let t = Mat4::from_translation(Vec3::new(0.0, joint.length, 0.0));

			accumulator = accumulator * r * t;
			models.push(accumulator);
		}
		models
	}
}

pub fn rad(degrees: f32) -> f32 {
	PI * degrees / 180.0
}

// right handed
// azimuth from z to x
// elevation from y
fn to_view_direction(azimuth: f32, elevation: f32) -> Vec3 {
	Vec3::new(
		azimuth.sin() * elevation.sin(),
		elevation.cos(),
		azimuth.cos() * elevation.sin(),
	)
}

fn dome_point(radius: f32) -> Vec3 {
	let mut rng = rand::thread_rng();

	let unit = to_view_direction(
		rng.gen_range(0.0..PI),
		rng.gen_range(-PI * 0.5..PI * 0.5)
	);
	unit * rng.gen_range(radius * 0.3..radius * 0.8)
}

pub fn chain_system(
	mut q_chain: Query<(&mut Chain, &Transform), Without<IsTarget>>,
	mut q_target: Query<&mut Transform, With<IsTarget>>
) {
	let target = dome_point(9.0);

	for ((mut chain, &chain_transform), mut target_transform) in zip(q_chain.iter_mut(), q_target.iter_mut()) {
		match &mut chain.state {
			ChainState::Done => {
				let mut target_angles = (chain.ik_fn)(&chain, target);

				for (start, target_angle) in zip(&chain.angles, &mut target_angles) {
					let difference = *target_angle - start;
					if difference.abs() > 180.0 {
						*target_angle -= 360.0 * (difference / 360.0).round();
					}
				}

				target_transform.translation = chain_transform * target;

				chain.state = ChainState::Seeking {
					target,
					transition_frames: 180,
					current_frame: 0,
					start_angles: chain.angles.to_vec(),
					target_angles,
				};
			},
			ChainState::Seeking { target_angles, transition_frames, current_frame, start_angles, .. } => {
				*current_frame += 1;

				if current_frame == transition_frames {
					chain.state = ChainState::Waiting { wait_frames: 60 }
				} else {
					let t = (*current_frame + 1) as f32 / (*transition_frames) as f32; // TODO: why +1

					let angles: Vec<f32> = zip(start_angles, target_angles).map(
						|(&mut start, &mut target)| { start + t * (target - start) }
					).collect();

					chain.angles = angles;
				}
			},
			ChainState::Waiting { wait_frames } => {
				if *wait_frames == 0 {
					chain.state = ChainState::Done;
				} else {
					*wait_frames -= 1;
				}
			},
		}
	}
}

pub fn chain_mesh_system(q_chain: Query<(&Chain, &Children)>, mut q_joints: Query<&mut Transform, With<IsJoint>>) {
	for (chain, children) in q_chain.iter() {
		let mut cumulative_transform = Transform::identity();

		for (joint, &angle, &child) in multizip((&chain.joints, &chain.angles, children.iter())) {
			let mut transform = q_joints.get_mut(child).expect("Unable to get joint child");

			let rotation = Transform::from_rotation(Quat::from_axis_angle(joint.axis.to_vec3(), rad(angle)));
			let translation = Transform::from_translation(Vec3::new(0.0, joint.length, 0.0));

			*transform = cumulative_transform * rotation;

			cumulative_transform = *transform * translation;
		}
	}
}

pub fn setup_chains(
	mut commands: Commands,
	mut meshes: ResMut<Assets<Mesh>>,
	mut materials: ResMut<Assets<StandardMaterial>>,
) {
	let make_chain = &mut |position: Vec3, ik_fn: fn(&Chain, Vec3)->Vec<f32>| {
		const CHAINS_COLOR: Color = Color::rgb(0.8, 0.7, 0.6);
		const WIDTH: f32 = 0.8;
		const LENGTH: f32 = 3.0;

		let joint_pbr: PbrBundle = PbrBundle {
			mesh: meshes.add(Mesh::from(shape::Box {
				min_x: -WIDTH / 2.0,
				max_x:  WIDTH / 2.0,
				min_y: 0.0,
				max_y: LENGTH,
				min_z: -WIDTH / 2.0,
				max_z:  WIDTH / 2.0,
			})),
			material: materials.add(CHAINS_COLOR.into()),
			..Default::default()
		};
		let joints = vec![
			Joint { axis: Axis::Y, length: 0.0,          },
			Joint { axis: Axis::X, length: LENGTH as f32 },
			Joint { axis: Axis::X, length: LENGTH as f32 },
			Joint { axis: Axis::X, length: LENGTH as f32 },
		];
		let angles = vec![0.0, 0.0, 0.0, 0.1];

		const PEDESTAL_SIZE: f32 = 2.0;
		commands.spawn_bundle(PbrBundle {
			mesh: meshes.add(Mesh::from(shape::Cube::new(PEDESTAL_SIZE))),
			transform: Transform::from_xyz(position.x, position.y + PEDESTAL_SIZE / 2.0, position.z),
			material: materials.add(CHAINS_COLOR.into()),
			..Default::default()
		});

		let chain_id = commands
			.spawn()
			.insert(Chain{ joints, angles, ik_fn, state: ChainState::default() })
			.insert(Transform::from_xyz(position.x, position.y + PEDESTAL_SIZE, position.z))
			.insert(GlobalTransform::default())
			.id();

		let j0_id = commands
			.spawn()
			.insert(Transform::default())
			.insert(GlobalTransform::default())
			.insert(IsJoint)
			.id();

		let j1_id = commands
			.spawn_bundle(joint_pbr.clone())
			.insert(IsJoint)
			.id();

		let j2_id = commands
			.spawn_bundle(PbrBundle {
				transform: Transform::from_xyz(0.0, LENGTH, 0.0),
				..joint_pbr.clone()})
			.insert(IsJoint)
			.id();

		let j3_id = commands
			.spawn_bundle(PbrBundle {
				transform: Transform::from_xyz(0.0, LENGTH, 0.0),
				..joint_pbr})
			.insert(IsJoint)
			.id();

		commands.entity(chain_id).push_children(&[j0_id, j1_id, j2_id, j3_id]);

		commands.spawn_bundle(PbrBundle {
			mesh: meshes.add(Mesh::from(shape::Cube::new(1.0))),
			..Default::default()
		}).insert(IsTarget);
	};

	make_chain(Vec3::new(-10.0, 0.0, -10.0), cyclic_coordinate_descent);
	make_chain(Vec3::new(0.0, 0.0, 0.0), jacobian_pseudo_inverse);
	make_chain(Vec3::new(10.0, 0.0, 10.0), jacobian_transpose);
}