use bevy::math::{Mat4, Vec3, Vec4};
use itertools::zip;
use nalgebra::{DVector, Dynamic, Matrix, U3, VecStorage};
use nalgebra::Vector3 as NALGVector3;
use crate::Chain;


type JMatrix = Matrix<f32, U3, Dynamic, VecStorage<f32, U3, Dynamic>>;

pub fn jacobian_pseudo_inverse(chain: &Chain, target: Vec3) -> Vec<f32> {
	const DISTANCE_THRESHOLD: f32 = 0.01;
	const MAX_ITERATIONS: usize = 240;

	if chain.joints.is_empty() { return Vec::new() }

	let mut angles: DVector<f32> = DVector::from_column_slice(&chain.angles);

	for _ in 0..MAX_ITERATIONS {
		let mut cumulative_transforms = vec![Mat4::IDENTITY];
		cumulative_transforms.extend(chain.cumulative_transforms_with_angles(angles.as_slice()).into_iter());

		let mut jacobian = JMatrix::from_element(chain.joints.len(), 0.0);

		let end = (*cumulative_transforms.last().unwrap() * Vec4::W).truncate();

		if (end - target).length() < DISTANCE_THRESHOLD { break }

		for (idx, (transform, joint)) in zip(cumulative_transforms, &chain.joints).enumerate() {
			let position = (transform * Vec4::W).truncate();

			let axis = (transform * joint.axis.to_vec3().extend(0.0)).truncate();

			let j = axis.cross(target - position);
			unsafe {
				*jacobian.get_unchecked_mut((0, idx)) = j.x;
				*jacobian.get_unchecked_mut((1, idx)) = j.y;
				*jacobian.get_unchecked_mut((2, idx)) = j.z;
			}
		}
		let desired_change = target - end;

		let dc: NALGVector3<f32> = NALGVector3::new(desired_change.x, desired_change.y, desired_change.z);

		let transpose = jacobian.transpose();

		let inverse = (jacobian * transpose.clone()).try_inverse().unwrap();

		let pseudo_inverse = transpose * inverse;

		angles += pseudo_inverse * dc;
	}
	Vec::from(angles.as_slice())
}
