use bevy::math::{Mat4, Vec3, Vec4};
use crate::{Chain, chain::rad};


pub fn cyclic_coordinate_descent(chain: &Chain, target: Vec3) -> Vec<f32> {
	const DISTANCE_THRESHOLD: f32 = 0.01;
	const PERP_LENGTH_THRESHOLD: f32 = 0.0001;
	const MAX_ITERATIONS: usize = 240;

	if chain.joints.is_empty() { return Vec::new() }

	let len = chain.joints.len();

	let mut cumulative_transforms = chain.cumulative_transforms();

	let mut angles: Vec<f32> = chain.angles.to_vec();

	let mut end = (*cumulative_transforms.last().unwrap() * Vec4::W).truncate();

	for _ in 0..MAX_ITERATIONS {
		if (end - target).length() < DISTANCE_THRESHOLD { break }

		let mut reverse_accumulator = Mat4::IDENTITY;

		for i in (0..len).rev() {
			let (joint, prev_transform, current_transform, angle) = unsafe {
				(
					chain.joints.get_unchecked(i),
					if i > 0 { *cumulative_transforms.get_unchecked(i - 1) }
					else     { Mat4::IDENTITY },
					cumulative_transforms.get_unchecked_mut(i),
					angles.get_unchecked_mut(i),
				)
			};
			let root = (prev_transform * Vec4::W).truncate();
			let axis = (prev_transform * joint.axis.to_vec3().extend(0.0)).truncate();

			let target_dir = target - root;
			let target_perp = target_dir - (axis * axis.dot(target_dir));
			let target_perp_len = target_perp.length();

			if target_perp_len < PERP_LENGTH_THRESHOLD { continue }

			let target_perp_norm = target_perp / target_perp_len;

			let end_dir = end - root;
			let end_perp = end_dir - (axis * axis.dot(end_dir));
			let end_perp_len = end_perp.length();

			if end_perp_len < PERP_LENGTH_THRESHOLD { continue }

			let end_perp_norm = end_perp / end_perp_len;

			let cross = end_perp_norm.cross(target_perp_norm);
			let magnitude = cross.length();
			let sign = if cross.dot(axis) > 0.0 { 1.0 } else { -1.0 };
			*angle += sign * magnitude.asin();

			// TODO: this assumption about Y being our offset direction needs to be put in a single place
			//
			let translation = Mat4::from_translation(Vec3::new(0.0, joint.length, 0.0));
			let rotation    = Mat4::from_axis_angle(joint.axis.to_vec3(), rad(*angle));
			let transform   = rotation * translation;

			*current_transform = prev_transform * transform;
			reverse_accumulator = transform * reverse_accumulator;

			let total = prev_transform * reverse_accumulator;
			end = (total * Vec4::W).truncate();
		}
	}
	angles
}
