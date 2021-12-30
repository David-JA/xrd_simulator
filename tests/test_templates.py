import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from xrd_simulator import templates, utils
import matplotlib.pyplot as plt

class TestUtils(unittest.TestCase):

    def setUp(self):
        np.random.seed(5) # changes all randomization in the test

    def test_s3dxrd(self):

        parameters = {
            "detector_distance"             : 191023.9164,
            "detector_center_pixel_z"       : 256.2345,
            "detector_center_pixel_y"       : 255.1129,
            "pixel_side_length_z"           : 181.4234,
            "pixel_side_length_y"           : 180.2343,
            "number_of_detector_pixels_z"   : 512,
            "number_of_detector_pixels_y"   : 512,
            "wavelength"                    : 0.285227,
            "beam_side_length_z"            : 512 * 200.,
            "beam_side_length_y"            : 512 * 200.,
            "rotation_step"                 : np.radians(1.634),
            "rotation_axis"                 : np.array([0., 0., 1.0])
        }

        beam, detector, motion = templates.s3dxrd( parameters )
        for ci in beam.centroid:
            self.assertAlmostEqual(ci, 0, msg="beam not at origin.")

        det_approx_centroid = detector.d0.copy()
        det_approx_centroid[1] += detector.d1[1]
        det_approx_centroid[2] += detector.d2[2]

        self.assertAlmostEqual(det_approx_centroid[0], parameters["detector_distance"], msg="Detector distance wrong.")
        self.assertLessEqual(np.abs(det_approx_centroid[1]), 5*parameters["pixel_side_length_y"], msg="Detector not centered.")
        self.assertLessEqual(np.abs(det_approx_centroid[2]), 5*parameters["pixel_side_length_z"], msg="Detector not centered.")

        original_vector = np.random.rand(3,)-0.5
        time = 0.234986
        transformed_vector = motion( original_vector, time )

        angle = parameters["rotation_step"] * time
        s,c = np.sin(angle),np.cos(angle)
        Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])

        self.assertAlmostEqual(transformed_vector[0], np.dot(Rz,original_vector)[0], msg="Motion does not rotate around z-axis")
        self.assertAlmostEqual(transformed_vector[1], np.dot(Rz,original_vector)[1], msg="Motion does not rotate around z-axis")
        self.assertAlmostEqual(transformed_vector[2], original_vector[2], msg="Motion does not rotate around z-axis")


    def test_polycrystal_from_odf(self):

        unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.]
        sgname = 'P3221' # Quartz
        orientation_density_function = lambda x,q: 1./(np.pi**2) # uniform ODF
        number_of_crystals = 500
        sample_bounding_cylinder_height = 50
        sample_bounding_cylinder_radius = 25
        maximum_sampling_bin_seperation = np.radians(10.0)
        strain_tensor = lambda x: np.array([[0,0, 0.02*x[2]/sample_bounding_cylinder_height],[0,0,0],[0,0,0]]) # Linear strain gradient along rotation axis.

        polycrystal = templates.polycrystal_from_odf(  orientation_density_function,
                                                       number_of_crystals,
                                                       sample_bounding_cylinder_height,
                                                       sample_bounding_cylinder_radius,
                                                       unit_cell,
                                                       sgname,
                                                       maximum_sampling_bin_seperation,
                                                       strain_tensor )

        # Compare Euler angle distributions to scipy random uniform orientation sampler
        euler1 = np.array([ Rotation.from_matrix( U ).as_euler( 'xyz', degrees=True ) for U in polycrystal.eU_lab])
        euler2 = Rotation.random( 10 * euler1.shape[0] ).as_euler('xyz')

        for i in range(3):
            hist1, bins = np.histogram(euler1[:,i])
            hist2, bins = np.histogram(euler2[:,i])
            hist2 = hist2 / 10.
            # These histograms should look roughly the same
            self.assertLessEqual( np.max(np.abs(hist1 - hist2)), number_of_crystals*0.05, "ODF not sampled correctly." )

        parameters = {
            "detector_distance"             : 191023.9164,
            "detector_center_pixel_z"       : 256.2345,
            "detector_center_pixel_y"       : 255.1129,
            "pixel_side_length_z"           : 181.4234,
            "pixel_side_length_y"           : 180.2343,
            "number_of_detector_pixels_z"   : 512,
            "number_of_detector_pixels_y"   : 512,
            "wavelength"                    : 0.285227,
            "beam_side_length_z"            : 512 * 200.,
            "beam_side_length_y"            : 512 * 200.,
            "rotation_step"                 : np.radians(20.0),
            "rotation_axis"                 : np.array([0., 0., 1.0])
        }

        beam, detector, motion = templates.s3dxrd( parameters )

        number_of_crystals = 100
        sample_bounding_cylinder_height = 256 * 180 / 128.
        sample_bounding_cylinder_radius = 256 * 180 / 128.

        polycrystal = templates.polycrystal_from_odf( orientation_density_function,
                                                        number_of_crystals,
                                                        sample_bounding_cylinder_height,
                                                        sample_bounding_cylinder_radius,
                                                        unit_cell,
                                                        sgname,
                                                        maximum_sampling_bin_seperation,
                                                        strain_tensor )

        polycrystal.diffract( beam, detector, motion, min_bragg_angle=0, max_bragg_angle=None, verbose=True )
        diffraction_pattern = detector.render(frame_number=0,lorentz=False, polarization=False, structure_factor=False, method="centroid", verbose=True)
        bins, histogram = utils.diffractogram(diffraction_pattern>1, parameters['detector_center_pixel_z'], parameters['detector_center_pixel_y'])
        plt.plot(bins, histogram)
        plt.show()

        plt.imshow( diffraction_pattern )
        plt.show()

    def test_get_uniform_powder_sample(self):
        sample_bounding_radius = 1.203
        polycrystal = templates.get_uniform_powder_sample(
                        sample_bounding_radius = sample_bounding_radius,
                        number_of_grains = 15,
                        unit_cell = [4.926, 4.926, 5.4189, 90., 90., 120.],
                        sgname = 'P3221'
                        )
        for c in polycrystal.mesh_lab.coord:
            self.assertLessEqual( np.linalg.norm(c), sample_bounding_radius+1e-8, msg='Powder sample not contained by bounding sphere.' )


if __name__ == '__main__':
    unittest.main()