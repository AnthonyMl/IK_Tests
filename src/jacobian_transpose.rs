use bevy::math::{Mat4, Vec3, Vec4};
use itertools::zip;
use crate::Chain;


pub fn jacobian_transpose(chain: &Chain, target: Vec3) -> Vec<f32> {
	const DISTANCE_THRESHOLD: f32 = 0.01;
	const MAX_ITERATIONS: usize = 120;

	if chain.joints.is_empty() { return Vec::new() }

	let mut angles = chain.angles.clone();

	for _ in 0..MAX_ITERATIONS {
		let mut cumulative_transforms = vec![Mat4::IDENTITY];
		cumulative_transforms.extend(chain.cumulative_transforms_with_angles(&angles).into_iter());

		let end = (*cumulative_transforms.last().unwrap() * Vec4::W).truncate();

		if (end - target).length() < DISTANCE_THRESHOLD { break }

		let desired_change = target - end;

		for (idx, (transform, joint)) in zip(cumulative_transforms, &chain.joints).enumerate() {
			let position = (transform * Vec4::W).truncate();

			let axis = (transform * joint.axis.to_vec3().extend(0.0)).truncate();

			let j = axis.cross(target - position);

			angles[idx] += j.dot(desired_change);
		}
	}
	angles
}
