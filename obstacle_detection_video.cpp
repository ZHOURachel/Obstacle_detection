#include <iostream>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>

#include <Eigen/Dense>
#include <boost/asio.hpp>
#include <boost/bind.hpp>

#include <fstream>
#include <cmath>
#include <memory.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
// the threshold for detecte the plane
double threshold = 0.05;
// the maximun iteration time for RANSAC
size_t maxIteration = 1000;

double robot_height = 0.4;

// PCL visualizer
pcl::visualization::PCLVisualizer viewer("PCL Viewer");


// the spatial resolution for the gride map
double spatial_resolution = 0.05;
int matrix_size = 20 / spatial_resolution;

// the number of the gride in length
const int numGrideLength = 0.8 / spatial_resolution;
// the number of the gride in width
const int numGrideWidth = 0.25 / spatial_resolution;

// the parametres for sending the velocity to the lower machine
unsigned char startByte = 0xff;
unsigned char endByte = 0xfe;
std::string Base_Port = "/dev/ttyUSB0";

boost::asio::io_service iosev;
boost::asio::serial_port sp(iosev);


const float v = 0.3;
const float w = 0.5;
int isStuck = 0;
int turnDirection = 0;
int oldDirection = 0;

// Mutex //
boost::mutex cloud_mutex;

bool isFirstFrame = true;

// the structure to stock the occupancy map
struct OccupancyMap
{
	Eigen::MatrixXi coordinatePositive;
	Eigen::MatrixXi coordinateNegative;
	Eigen::MatrixXi pointsNumCounterPositive;
	Eigen::MatrixXi pointsNumCounterNegative;
} occupancy_map = {Eigen::MatrixXi::Zero(matrix_size, matrix_size),
                   Eigen::MatrixXi::Zero(matrix_size, matrix_size),
                   Eigen::MatrixXi::Zero(matrix_size, matrix_size),
                   Eigen::MatrixXi::Zero(matrix_size, matrix_size)
                  };

// struct OccupancyMap
// {
// 	int coordinatePositive[500];
// 	int coordinateNegative[500];
// 	int pointsNumCounterPositive[500];
// 	int pointsNumCounterNegative[500];
// } occupancy_map ;


// the structure to stock the velocity
union Max_Value {
	unsigned char buf[8];
	struct _FLOAT_ {
		float _double_vT;
		float _double_vR;
	} Double_RAM;
} Send_Data;

void cloud_cb_ (const PointCloudT::ConstPtr &callback_cloud, PointCloudT::Ptr& cloud, bool* new_cloud_available_flag )
{
	cloud_mutex.lock ();    // in case of overwriting the current point cloud by another thread
	*cloud = *callback_cloud;
	*new_cloud_available_flag = true;
	cloud_mutex.unlock ();
}

/**
 * reset all the values of the occupancy map to 0
 */
void reinitOccupancyMap()
{
	occupancy_map.coordinatePositive.topLeftCorner(occupancy_map.coordinatePositive.rows(), occupancy_map.coordinatePositive.cols())
	    = Eigen::MatrixXi::Zero(occupancy_map.coordinatePositive.rows(), occupancy_map.coordinatePositive.cols());
	occupancy_map.coordinateNegative.topLeftCorner(occupancy_map.coordinateNegative.rows(), occupancy_map.coordinateNegative.cols())
	    = Eigen::MatrixXi::Zero(occupancy_map.coordinateNegative.rows(), occupancy_map.coordinateNegative.cols());
	occupancy_map.pointsNumCounterPositive.topLeftCorner(occupancy_map.pointsNumCounterPositive.rows(), occupancy_map.pointsNumCounterPositive.cols())
	    = Eigen::MatrixXi::Zero(occupancy_map.pointsNumCounterPositive.rows(), occupancy_map.pointsNumCounterPositive.cols());
	occupancy_map.pointsNumCounterNegative.topLeftCorner(occupancy_map.pointsNumCounterNegative.rows(), occupancy_map.pointsNumCounterNegative.cols())
	    = Eigen::MatrixXi::Zero(occupancy_map.pointsNumCounterNegative.rows(), occupancy_map.pointsNumCounterNegative.cols());

	// memset(&occupancy_map, 0, sizeof(occupancy_map));
}

// void reinitOccupancyMap()
// {
// 	memset(occupancy_map.coordinatePositive, 0, sizeof(occupancy_map.coordinatePositive));
// 	memset(occupancy_map.coordinateNegative, 0, sizeof(occupancy_map.coordinateNegative));
// 	memset(occupancy_map.pointsNumCounterPositive, 0, sizeof(occupancy_map.pointsNumCounterPositive));
// 	memset(occupancy_map.pointsNumCounterNegative, 0, sizeof(occupancy_map.pointsNumCounterNegative));
// }

/**
 * open the port in order to send the velocity to the lower machine
 */
void openPort()
{
	sp.open(Base_Port);
	sp.set_option(boost::asio::serial_port::baud_rate(9600));
	sp.set_option(boost::asio::serial_port::flow_control(boost::asio::serial_port::flow_control::none));
	sp.set_option(boost::asio::serial_port::parity(boost::asio::serial_port::parity::none));
	sp.set_option(boost::asio::serial_port::stop_bits(boost::asio::serial_port::stop_bits::one));
	sp.set_option(boost::asio::serial_port::character_size(8));
}

/**
 * send the velocity to the lower machine
 * @param vTranslation [the translation velocity]
 * @param vRotation    [the rotation velocity]
 */
void sendVelocity(float vTranslation, float vRotation)
{

	Send_Data.Double_RAM._double_vT = vTranslation;
	Send_Data.Double_RAM._double_vR = vRotation;

	write(sp, boost::asio::buffer(&startByte, 1));
	write(sp, boost::asio::buffer(Send_Data.buf, 8));
	write(sp, boost::asio::buffer(&endByte, 1));

	//iosev.run();

}


void passthroughFilter( PointCloudT::Ptr& cloud, PointCloudT::Ptr& cloud_filtered, float filterLimit )
{
	pcl::PassThrough<PointT> pass;
	pass.setInputCloud(cloud);
	pass.setFilterFieldName("z");
	pass.setFilterLimits(0.0, filterLimit);

	pass.filter(*cloud_filtered);
}

/**
 * segmente the plane from the point cloud, and  also calculate the normal of the cloud before segmentation as well as the plane coefficients
 * @param  cloud               [the the inpute point cloud ]
 * @param  coefficients_plane  [the coefficients of the plane]
 * @param  cloud_normals       [the normals of the cloud]
 * @param  maxIteration        [the number of iteration for the RANSAC algo]
 * @param  threshold           [the threshold for the RANSAC algo]
 * @return                     [the index of the plane points in the cloud]
 */

pcl::PointIndices::Ptr plane_segmentation( pcl::PointCloud<PointT>::Ptr cloud,
        pcl::ModelCoefficients::Ptr coefficients_plane,
        size_t maxIteration, double threshold)
{
	pcl::NormalEstimation<PointT, pcl::Normal> ne;
	pcl::SACSegmentationFromNormals <PointT, pcl::Normal> seg;
//  pcl::ExtractIndices<PointT> extract;
	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
	// pcl::visualization::PCLVisualizer * viewer (new pcl::visualization::PCLVisualizer("PCL Visualizer"));

	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

	pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices);

	ne.setSearchMethod (tree);
	ne.setInputCloud (cloud);
	ne.setKSearch (50);
	ne.compute (*cloud_normals);

	seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
	seg.setNormalDistanceWeight (0.1);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setMaxIterations (maxIteration);
	seg.setDistanceThreshold (threshold);
	seg.setInputCloud (cloud);
	seg.setInputNormals (cloud_normals);

	seg.segment (*inliers_plane, *coefficients_plane);
	// std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

	if (inliers_plane->indices.size () == 0)
	{
		PCL_ERROR ("Could not estimate a planar model for the given dataset.");
		// exit (0);
	}

	return inliers_plane;
}

/**
 * verify if the plane selected is correct or not, here we consider the standard plane model is 0x - y + 0z + height_robot = 0
 * this function will calculate the angle between the normal vectors of the select plane and the standard plane, which the normal vector is (0, 1, 0)
 * @param  coefficients_plane [the coefficients of the select plane model, here the type is pcl::ModelCoefficients::Ptr]
 * @return                    [true if the selected plane is correct, false if not]
 */
bool verifyPlaneModel(pcl::ModelCoefficients::Ptr coefficients_plane)
{
	double cosTheta = ( coefficients_plane->values[1] * 1 ) /
	                  ( sqrt( coefficients_plane->values[0] * coefficients_plane->values[0]
	                          + coefficients_plane->values[1] * coefficients_plane->values[1]
	                          + coefficients_plane->values[2] * coefficients_plane->values[2])
	                    * sqrt( 1 * 1 ) );
	double theta =  acos(cosTheta) * 180 / 3.1415 ;
	// std::cout << "cosTheta = " << cosTheta << "		" << "theta = " << acos(cosTheta) << std::endl;
	// std::cout << "theta= " << theta << std::endl;
	if ( theta > 15 && theta < 165)
		return false;
	else
		return true;
}


/**
 * verify if the plane selected is correct or not, here we consider the standard plane model is 0x - y + 0z + height_robot = 0
 * this function will calculate the angle between the normal vectors of the select plane and the standard plane, which the normal vector is (0, 1, 0)
 * @param  coefficients_plane [the coefficients of the select plane model, here the type is Eigen::VectorXf]
 * @return                    [true if the selected plane is correct, false if not]
 */
bool verifyPlaneModel(Eigen::VectorXf& coefficients_plane)
{
	double cosTheta = (coefficients_plane[1]) /
	                  ( sqrt( coefficients_plane[0] * coefficients_plane[0]
	                          + coefficients_plane[1] * coefficients_plane[1]
	                          + coefficients_plane[2] * coefficients_plane[2])
	                    * sqrt( 1 * 1 ) );
	double theta = acos(cosTheta) * 180 / 3.1415 ;

	// double theta = (acos(cosTheta) * 180 / 3.1415)  > 90 ?  ( 180 - acos(cosTheta) * 180 ) : ( acos(cosTheta) * 180 / 3.1415 ) ;

	// std::cout << "cosTheta = " << cosTheta << "		" << "theta = " << acos(cosTheta) << std::endl;
	// std::cout << "theta = " << theta << std::endl;
	if ( theta > 15 && theta < 165)
		return false;
	else
		return true;
}

/**
 * recompute the plane model coefficients with the ransac algorithme
 * @param cloud              [the input cloud]
 * @param coefficients_plane [the current plane coefficients, after this function, the coefficients will be updated]
 * @param maxIteration       [the max number of iterations for the algorithme]
 * @param threshold          [the max distance between the points and the selected plane]
 * @param inliers_plane      [the inliers points for the plane model]
 */
void reComputePlaneModelCoefficients( pcl::PointCloud<PointT>::Ptr cloud,
                                      Eigen::VectorXf&  coefficients_plane,
                                      size_t maxIteration, double threshold,
                                      std::vector<int>& inliers_plane)
{
	pcl::SampleConsensusModelPlane <PointT>::Ptr model_plane(new pcl::SampleConsensusModelPlane<PointT> (cloud));
	pcl::RandomSampleConsensus<PointT> ransac (model_plane);
	ransac.setDistanceThreshold (threshold);
	ransac.setMaxIterations(maxIteration);
	ransac.computeModel();
	ransac.getInliers(inliers_plane);

	if (inliers_plane.size () == 0)
	{
		PCL_ERROR ("Could not estimate a planar model for the given dataset.");
	}
	else
	{
		ransac.getModelCoefficients(coefficients_plane);
	}

}

/**
 * reset the plane coefficients by the standard plane coefficients, which is [0, -1, 0, robot_height]
 * @param coefficients_plane [the current plane coefficients, after this function, the coefficients will be the standard plane coefficients]
 */
void resetPlaneModel(pcl::ModelCoefficients::Ptr coefficients_plane)
{
	coefficients_plane->values[0] = 0;
	coefficients_plane->values[1] = -1;
	coefficients_plane->values[2] = 0;
	coefficients_plane->values[3] = robot_height;
}

/**
 * reset the plane coefficients by the standard plane coefficients, which is [0, -1, 0, robot_height]
 * @param coefficients_plane [the current plane coefficients, after this function, the coefficients will be the standard plane coefficients]
 */
void resetPlaneModel(Eigen::VectorXf& coefficients_plane)
{
	coefficients_plane.resize(4);
	coefficients_plane[0] = 0;
	coefficients_plane[1] = -1;
	coefficients_plane[2] = 0;
	coefficients_plane[3] = robot_height;
	// std::cout << "after reset" << std::endl << coefficients_plane << std::endl;
}

/**
 * extract the points of the plane into the cloud_plane, with the index in the PointIndices
 * @param  cloud         [the input point cloud to extract from]
 * @param  inliers_plane [the index of the plane points, it is the PointIndices type]
 * @return               [the plane cloud]
 */
pcl::PointCloud<PointT>::Ptr extractPlane ( PointCloudT::Ptr cloud, pcl::PointIndices::Ptr inliers_plane )
{
	// std::cout << "extract plane" << std::endl;
	pcl::ExtractIndices<PointT> extract;
	extract.setInputCloud (cloud);
	extract.setIndices (inliers_plane);
	extract.setNegative (false);

	pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
	extract.filter (*cloud_plane);
	// std::cerr << "The plane cloud has " << cloud_plane->size() << " points " << std::endl;
	return cloud_plane;
}

/**
 * exteract the points of the plane into the cloud_plane, with the index in the vector
 * @param  cloud         [the input cloud]
 * @param  inliers_plane [the index of the plane points, it is the vector type]
 * @return               [the plane cloud]
 */
pcl::PointCloud<PointT>::Ptr extractPlane ( PointCloudT::Ptr cloud, std::vector<int> inliers_plane )
{
	// std::cout << "extract plane" << std::endl;

	pcl::PointIndices::Ptr indices_plane (new pcl::PointIndices );

	indices_plane->indices = inliers_plane;
	pcl::ExtractIndices<PointT> extract;
	extract.setInputCloud (cloud);
	extract.setIndices (indices_plane);
	extract.setNegative (false);

	pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
	extract.filter (*cloud_plane);
	// std::cerr << "The plane cloud has " << cloud_plane->size() << " points " << std::endl;
	return cloud_plane;
}

/**
 * find all the points belong to the plane and return the index of the points
 * @param  model_coefficients         [the coefficients of the plane]
 * @param  threshold                  [the threshold of the distance between the point and the plane]
 * @param  cloud                      [the inpute point cloud]
 * @return                            [the index of the planepoints]
 */
std::vector<int> findPlanePoint (Eigen::VectorXf model_coefficients, double threshold, PointCloudT::Ptr cloud)
{
	std::vector<int> inliers_plane;

	pcl::SampleConsensusModelPlane <PointT>::Ptr plane(new pcl::SampleConsensusModelPlane<PointT> (cloud));

	plane->selectWithinDistance (model_coefficients, threshold, inliers_plane );

	return inliers_plane;

}

/**
 * calibrate the coefficients of the plane, the model_coefficients are update after calibrated
 * if the coefficient can't be calibrated, the model_coefficients doesn't change
 * @param cloud              [the inpute point cloud]
 * @param inliers_plane      [the index of the plane points]
 * @param model_coefficients [the old coefficients of the plane]
 */
void calibratePlaneCoefficients (PointCloudT::Ptr cloud,
                                 const std::vector<int>& inliers_plane,
                                 Eigen::VectorXf& model_coefficients)
{
	pcl::SampleConsensusModelPlane<PointT>::Ptr
	model_p (new pcl::SampleConsensusModelPlane<PointT> (cloud));

	Eigen::VectorXf new_coefficients;
	new_coefficients.resize(4);

	model_p->optimizeModelCoefficients ( inliers_plane, model_coefficients, new_coefficients);

	model_coefficients = new_coefficients;
}

/**
 * get the distance from the point to the plane  and return the distance
 * @param  model_coefficients [the coefficients of the plane]
 * @param  point              [the point to calculate the distance]
 * @return                    [the distance from the point to the distance]
 */
float getDistanceFromPlane (Eigen::VectorXf& model_coefficients, const PointT& point)
{
	return ((model_coefficients[0] * point.x )
	        + (model_coefficients[1] * point.y )
	        + (model_coefficients[2] * point.z )
	        + model_coefficients[3])
	       / sqrt(pow(model_coefficients[0], 2) + pow(model_coefficients[1], 2) + pow(model_coefficients[2], 2));
}


/**
 * remove the outliers of the cloud with the radius outlier removal filter and the statistical outlier removal filter
 * @param cloud          [the inpute cloud]
 * @param cloud_filtered [the cloud after filtered]
 */
void remove_outliers(PointCloudT::Ptr& cloud, PointCloudT::Ptr& cloud_filtered)
{

	// RadiusOutilierRemoval filter
	pcl::RadiusOutlierRemoval<PointT> outrem;
	// build the filter
	outrem.setInputCloud(cloud);
	outrem.setRadiusSearch(0.1);
	outrem.setMinNeighborsInRadius (20);
	// apply filter
	outrem.filter (*cloud_filtered);

	// //StatisticalOutlierRemoval filter
	// pcl::StatisticalOutlierRemoval<PointT> sor;
	// sor.setInputCloud (cloud_filtered);
	// sor.setMeanK (50);
	// sor.setStddevMulThresh (1.0);
	// sor.filter (*cloud_filtered);

}

/**
 * extract the obstacle points from the cloud and stock it into the obstacle_cloud
 * and it also returns a vector which is the distance for the points from the plane
 * @param  cloud              [the input point cloud to extract obstacle from]
 * @param  inliers_plane      [the index of the plane points ]
 * @param  model_coefficients [the coefficients of the plane]
 * @param  obstacle_cloud     [the obstacle cloud, the extract points are stocked in the obstacle_cloud]
 * @return                    [the distance between the points and the plane ]
 */

std::vector<float> extractObstaclePoints(PointCloudT::Ptr& cloud, pcl::PointIndices::Ptr& inliers_plane,
        Eigen::VectorXf& model_coefficients, PointCloudT::Ptr& obstacle_cloud)
{
	// std::cout << "extract obstacle " << std::endl;
	std::vector<float> height;
	size_t n = 0;

	for (size_t i = 0; i < cloud->size(); i++)
	{
		if ( i != inliers_plane->indices[n])
		{
			if (cloud->points[i].z <= 1.5)
			{
				float d = getDistanceFromPlane (model_coefficients, cloud->points[i]);
				if ( d < 1)
				{
					obstacle_cloud->push_back(cloud->points[i]);
					height.push_back(d);
				}
			}

		}
		else
			n++;
	}
	// std::cout << "obstacle cloud size = " << obstacle_cloud->size() << std::endl;
	return height;
}


/**
 * extract the obstacle points from the cloud and stock it into the obstacle_cloud
 * and it also returns a vector which is the distance for the points from the plane
 * @param  cloud              [the input point cloud to extract obstacle from]
 * @param  inliers_plane      [the index of the plane points ]
 * @param  model_coefficients [the coefficients of the plane]
 * @param  obstacle_cloud     [the obstacle cloud, the extract points are stocked in the obstacle_cloud]
 * @return                    [the distance between the points and the plane ]
 */

std::vector<float> extractObstaclePoints(PointCloudT::Ptr& cloud, std::vector<int>& inliers_plane,
        Eigen::VectorXf& model_coefficients, PointCloudT::Ptr& obstacle_cloud)
{
	// std::cout << "extract obstacle " << std::endl;

	std::vector<float> height;
	size_t n = 0;

	for (size_t i = 0; i < cloud->size(); i++)
	{
		if ( i != inliers_plane[n])
		{
			if (cloud->points[i].z <= 1.5)
			{
				float d = getDistanceFromPlane (model_coefficients, cloud->points[i]);
				if ( d < 1)
				{
					obstacle_cloud->push_back(cloud->points[i]);
					height.push_back(d);
				}
			}
		}
		else
			n++;
	}
	// std::cout << "obstacle cloud size = " << obstacle_cloud->size() << std::endl;

	return height;
}



/**
 * calculate the normals for the cloud
 * @param cloud         [the input cloud]
 * @param cloud_normals [the normals are stocked in the cloud_normals]
 */
void calculateNormals (PointCloudT::Ptr& cloud, pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals)
{
	pcl::NormalEstimation<PointT, pcl::Normal> ne;
	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());

	//pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

	ne.setSearchMethod (tree);
	if (cloud->points.size() == 0)
	{
		return;
	}
	else
	{
		ne.setInputCloud (cloud);
		ne.setKSearch (50);
		ne.compute (*cloud_normals);
	}
}


/**
 * get the norm of the vector between the "point" and the origin point
 * @param  point  [the current point]
 * @param  origin [the origin point]
 * @return        [the norm of the vector ||point - origin|| ]
 */
float getVectorNorm(const PointT &point, const PointT &origin)
{
	float d =  sqrt( (point.x - origin.x) * (point.x - origin.x) + (point.y - origin.y) * (point.y - origin.y) + (point.z - origin.z) * (point.z - origin.z) );
	// if (d == FLT_MAX)
	// {
	//   std::cout << point.x << "	" <<  origin.x << "		" << point.y << "	" << origin.y << "		" << point.z << "		" << origin.z << std::endl;
	// }
	return d;
}

/**
 * get the angle theta between point o and original point
 * @param  point  [the current point]
 * @param  origin [the origin point]
 * @return        [the value of the angle theta]
 */
float getTheta(const PointT &point, const PointT &origin)
{
	// std::cout << "point.x - origin.x = " << point.x << " - " << origin.x << " = " << point.x - origin.x << std::endl;
	// std::cout << "point.z - origin.z = " << point.z << " - " << origin.z << " = " << point.z - origin.z << std::endl;
	return atan( (point.x - origin.x) / (point.z - origin.z) );
}

/**
 * set the value of the gride correspondant to the gride(x, y)  to 1
 * @param x [the coordinate of the gride]
 * @param y [the coordinate of the gride]
 */
void setGrideValue ( int x, int y)
{
	if (x <= 0 )
	{
		std::cout << "occupancy map coordonate error! x = " << x << std::endl;
		sendVelocity(0, 0);
	}
	if (y == 0)
		y++;
	if (y > 0)
	{
		occupancy_map.coordinatePositive(x - 1, y - 1) = 1;
		// occupancy_map.coordinatePositive[x - 1, y - 1] = 1;
		occupancy_map.pointsNumCounterPositive (x - 1, y - 1) += 1;
		// occupancy_map.pointsNumCounterPositive [x - 1, y - 1] += 1;
	}
	else
	{
		occupancy_map.coordinateNegative(x - 1, abs(y) - 1) = 1;
		// occupancy_map.coordinateNegative[x - 1, abs(y) - 1] = 1;
		occupancy_map.pointsNumCounterNegative ( x - 1, abs(y) - 1 ) += 1;
		// occupancy_map.pointsNumCounterNegative [ x - 1, abs(y) - 1 ] += 1;
	}
}


/**
 * create the occupancy map and stock it in the struct OccupancyMap
 * @param cloud_obstacle     [the obstacle cloud]
 * @param height             [the height of the points from the plane]
 * @param plane_normal       [the normal of the plane]
 * @param model_coefficients [the coefficients of the plane]
 */
void createOccupancyMap ( PointCloudT::Ptr cloud_obstacle, const std::vector<float>& height,
                          pcl::Normal& plane_normal, Eigen::VectorXf& model_coefficients)
{
	// std::cout << "create the occupancy map! " << std::endl;
	PointT origin;
	origin.x = 0.f;
	origin.y = 0.f;
	origin.z = 0.f;

	float height_origin = getDistanceFromPlane(model_coefficients, origin);
	// std::cout << " the origin point height = " << height_origin << std::endl;

	origin.x = - height_origin * plane_normal.normal_x;
	origin.y = - height_origin * plane_normal.normal_y;
	origin.z = - height_origin * plane_normal.normal_z;

	for (size_t i = 0; i < cloud_obstacle->size(); i++)
	{
		PointT projectedPoint (origin);

		projectedPoint.x = cloud_obstacle->points[i].x - height[i] * plane_normal.normal_x;
		projectedPoint.y = cloud_obstacle->points[i].y - height[i] * plane_normal.normal_y;
		projectedPoint.z = cloud_obstacle->points[i].z - height[i] * plane_normal.normal_z;

		float d = 0.f;
		d = getVectorNorm(projectedPoint, origin);

		float theta = 0.f;
		theta = getTheta(projectedPoint, origin);

		float coordinate_x = d * cos(theta) / spatial_resolution;
		float coordinate_y = d * sin(theta) / spatial_resolution;
		if ( coordinate_x <= 0 )
		{
			std::cout << " d = " << d << std::endl;
			std::cout << "theta = " << theta << std::endl;
			std::cout << "there are errors in the projected points! projected x = " << coordinate_x << std::endl;
			sendVelocity(0, 0);
		}

		int x = ceil( coordinate_x ) ;

		int y;
		if (coordinate_y < 0 )
		{
			y = floor ( coordinate_y );
		}
		else
			y = ceil(coordinate_y);
		setGrideValue(x, y);
	}
}

/**
 * set the velocity by the occupancy map, it send the translation velocity or the rotation velocity to the lower machine
 */
void setVelocity()
{
	bool isLeftOccupied = false;
	bool isRightOccupied = false;
	int numLeftGride = 0;
	int numRightGride = 0;


	for (int i = 0; i < numGrideLength; i++)
	{
		for (int j = 0; j < numGrideWidth; j++)
		{
			if (occupancy_map.coordinatePositive(i, j) == 1)
			// if (occupancy_map.coordinatePositive[i, j] == 1)
			{
				isRightOccupied = true;
				numRightGride += 1;
			}
			if (occupancy_map.coordinateNegative(i, j) == 1)
			// if (occupancy_map.coordinateNegative[i, j] == 1)
			{
				isLeftOccupied = true;
				numLeftGride += 1;
			}
		}
	}

	float translation_v = 0.f;
	float rotation_v = 0.f;
	if (!isRightOccupied && !isLeftOccupied)     // there is nothing in the front
	{
		translation_v = v;
		isStuck = 0;
		turnDirection = 0;
	}
	else
	{
		if ( isStuck > 1 )   // if the robot is stuck in a corner where there are obstacles at both sides
		{
			rotation_v = turnDirection * w;
			std::cout << "Aiibot gets stuck!!!!!" << std::endl;
		}
		else
		{
			if (!isLeftOccupied && isRightOccupied)          // there are obstacles at the right side
			{
				rotation_v = w;
				turnDirection = 1;

			}
			else if (isLeftOccupied && !isRightOccupied)   // there are obstacles at the left side
			{
				rotation_v = w * (-1);
				turnDirection = -1;
			}
			else
			{
				if (numLeftGride > numRightGride)
				{
					rotation_v = w * (-1);
					turnDirection = -1;     // turn right
				}
				else
				{
					rotation_v = w;

					turnDirection = 1;    // turn left
				}
			}
		}
	}

	if (oldDirection * rotation_v < 0)
	{
		isStuck++;
	}
	oldDirection = turnDirection;

	std::cout << "left gride = " << numLeftGride << "   " << "right gride = " << numRightGride << std::endl;
	std::cout << "v = " << translation_v << "    " << "w = " << rotation_v << std::endl;
	sendVelocity(translation_v, rotation_v);
}

/**
 * get a unique normal vector of the cloud_normals, it calculate the average values of the 100 normals in middle of the cloud_normals
 * @param cloud_normals [the input cloud normals]
 * @param normalVector  [the normal vector to calculate]
 */
void getNormalVector(pcl::PointCloud<pcl::Normal>::Ptr& cloud_normals, pcl::Normal& normalVector)
{
	int counter = 0;
	for (size_t i = cloud_normals->size() / 2 - 50 ; i < cloud_normals->size() / 2 + 50 ; i++)
	{

		normalVector.normal_x += cloud_normals->points[i].normal_x;
		normalVector.normal_y += cloud_normals->points[i].normal_y;
		normalVector.normal_z += cloud_normals->points[i].normal_z;
		counter++;
	}

	normalVector.normal_x = normalVector.normal_x / counter;
	normalVector.normal_y = normalVector.normal_y / counter;
	normalVector.normal_z = normalVector.normal_z / counter;

}

/**
 * set the normal vector of a plane by the coefficients of the plane, for example, if a plane is describe by 4 coefficients: ax+by+cz+d=0,
 * then the normal vector of this plane is [a, b, c]
 * @param coefficients_plane [the coefficients of the plane, it contains 4 coefficients a, b, c and d] and the type is ModelCoefficients
 * @param normalVector       [the plane's normal vector to be set]
 */
void setPlaneNormalVector(pcl::ModelCoefficients::Ptr coefficients_plane, pcl::Normal& normalVector)
{
	normalVector.normal_x = coefficients_plane->values[0];
	normalVector.normal_y = coefficients_plane->values[1];
	normalVector.normal_z = coefficients_plane->values[2];
}

/**
 * set the normal vector of a plane by the coefficients of the plane, for example, if a plane is describe by 4 coefficients: ax+by+cz+d=0,
 * then the normal vector of this plane is [a, b, c]
 * @param coefficients_plane [the coefficients of the plane, it contains 4 coefficients a, b, c and d], and the type is Eigen::VectorXf
 * @param normalVector       [the plane's normal vector to be set]
 */
void setPlaneNormalVector(Eigen::VectorXf coefficients_plane, pcl::Normal& normalVector)
{
	normalVector.normal_x = coefficients_plane[0];
	normalVector.normal_y = coefficients_plane[1];
	normalVector.normal_z = coefficients_plane[2];
}

/**
 * print the minimum value of x, y and of the cloud resprectivement
 * @param cloud [the point cloud]
 */
void showMinValues(PointCloudT::Ptr &cloud)
{
	double min_x = 100000, min_y = 10000, min_z = 100000;
	for (int i = 0; i < cloud->size(); i++)
	{

		if (cloud->points[i].x < min_x)
			min_x = cloud->points[i].x;
		if (cloud->points[i].y < min_y)
			min_y = cloud->points[i].y;
		if (cloud->points[i].z < min_z)
			min_z = cloud->points[i].z;
	}
	std::cout << "min xyz = " << min_x << " " << min_y << " " << min_z << std::endl;
}


int main()
{

	PointCloudT::Ptr cloud (new PointCloudT);
	bool new_cloud_available_flag = false;
	pcl::Grabber* interface = new pcl::OpenNIGrabber ();
	boost::function <void (const PointCloudT::ConstPtr&)> f =
	    boost::bind (&cloud_cb_, _1, cloud, &new_cloud_available_flag);
	interface->registerCallback(f);
	interface->start();
	std::cerr << "The grabber started! " << std::endl;

	std::vector<int> indices_Nans;

	//  Wait for the first frame
	while (!new_cloud_available_flag)
		boost::this_thread::sleep(boost::posix_time::milliseconds(1));
	new_cloud_available_flag = false;

	// pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients);
	Eigen::VectorXf model_coefficients;
	model_coefficients.resize(4);
	model_coefficients << 1, 1, 1, 1;
	// std::cout<<model_coefficients<<std::endl;

	cloud_mutex.lock ();   // for not overwriting the point cloud

	pcl::removeNaNFromPointCloud(*cloud, *cloud, indices_Nans);

	// pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT> ());


	// std::cerr << "Point Cloud has: " << cloud->points.size () << " data points." << std::endl;

	// pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices );

	std::vector<int> inliers_plane;

	double start = pcl::getTime();
	// inliers_plane = plane_segmentation(cloud, coefficients_plane, maxIteration, threshold);  //segmentate the plane
	reComputePlaneModelCoefficients(cloud, model_coefficients, maxIteration, threshold, inliers_plane);
	double end = pcl::getTime();
	std::cout << "Segmente the plane use " << end - start << " s !" << std::endl;

	pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());


	cloud_plane = extractPlane ( cloud, inliers_plane );   //extract the plane cloud by the inliers_plane indices

	//to show the cloud (in RGB), the selected plane (in green) and the obstacle that should be dealed with (in red)
	pcl::visualization::PointCloudColorHandlerRGBField<PointT> cloudRGB (cloud);
	pcl::visualization::PointCloudColorHandlerCustom<PointT> planeGreen (cloud_plane, 0, 255, 0);
	viewer.addPointCloud<PointT>(cloud, cloudRGB, "original cloud");
	viewer.addPointCloud<PointT>(cloud_plane, planeGreen, "plane cloud");
	viewer.setCameraPosition ( 0, 0, -2, 0, -1, 0, 0 );

	// wait for the "q" is pressed
	viewer.spin();

	bool isStartPlaneRight = true;
	bool isValid =  verifyPlaneModel(model_coefficients);
	if (!isValid)
	{
		resetPlaneModel(model_coefficients);
		std::cout << "the plane coefficients are not valid! They have been reset! " << std::endl;
		isStartPlaneRight = false;
	}

	pcl::Normal plane_normalVector(0.f, 0.f, 0.f);
	setPlaneNormalVector(model_coefficients, plane_normalVector);

	cloud_mutex.unlock ();

	reinitOccupancyMap();

	// show the video in a new window
	pcl::visualization::PCLVisualizer viewer ("PCL Viewer");
	viewer.setCameraPosition ( 0, 0, -2, 0, -1, 0, 0 );

	// double time = pcl::getTime();

	// for the fps
	static unsigned count = 0;
	static double last = pcl::getTime();
	// openPort();
	while (!viewer.wasStopped())
	{
		// double startFrame = pcl::getTime();
		if (new_cloud_available_flag && cloud_mutex.try_lock ())
		{
			new_cloud_available_flag = false;

			std::vector<int> inliers_plane;

			pcl::removeNaNFromPointCloud(*cloud, *cloud, indices_Nans);
			// passthroughFilter(cloud, cloud, 2);
			// std::cout<<model_coefficients<<std::endl;
			if ( !verifyPlaneModel(model_coefficients) )
			{
				// sendVelocity(0, 0);
				reComputePlaneModelCoefficients(cloud, model_coefficients, maxIteration, threshold, inliers_plane);
				// resetPlaneModel(model_coefficients);
				std::cout << "Current plane model is wrong! reset the plane model" << std::endl;
				
				// viewer.spinOnce();
				// new_cloud_available_flag = true;
				// cloud_mutex.unlock ();
				// continue;
			}

			inliers_plane = findPlanePoint (model_coefficients, threshold, cloud);
			if (inliers_plane.size() == 0)
			{
				std::cout << "No points in the plane!!!" << std::endl;
				if ( !isStartPlaneRight)    // if aiibot is facing the wall or a big plane and it can see noting
				{
					std::cout << "Go back for a second!" << std::endl;
					// sendVelocity(-v, 0);	 // it will go back for 2 second
					sleep(1);
					std::cout << "stop and recompute the plane" << std::endl;
					// sendVelocity(0, 0);		 // then it will stop
					isStartPlaneRight = true;

				}
				else
				{
					std::cout << "recompute the plane model" << std::endl;
					// sendVelocity(0, 0);
					reComputePlaneModelCoefficients(cloud, model_coefficients, maxIteration, threshold, inliers_plane);
				}

				viewer.spinOnce();
				new_cloud_available_flag = true;
				cloud_mutex.unlock ();
				// double time = pcl::getTime();
				continue;
			}


			//calculte the plane normal vector for every frame , maybe this is not necessary
			cloud_plane = extractPlane ( cloud, inliers_plane );

			pcl::PointCloud<PointT>::Ptr cloud_obstacle (new pcl::PointCloud<PointT> ());

			std::vector<float> height;
			// double startExtractOb = pcl::getTime();
			height = extractObstaclePoints(cloud, inliers_plane, model_coefficients, cloud_obstacle);
			// std::cout << "Extract obstacle point use " << pcl::getTime() - startExtractOb << " s" << std::endl;

			// std::cout << "The obstacle cloud size = " << cloud_obstacle->size() << std::endl;
			pcl::removeNaNFromPointCloud(*cloud_obstacle, *cloud_obstacle, indices_Nans);

			// double startCreateMap = pcl::getTime();
			createOccupancyMap(cloud_obstacle, height, plane_normalVector, model_coefficients);
			// std::cout << "Create occupancy map use " << pcl::getTime() - startCreateMap << " s" << std::endl;

			// setVelocity();
			// double startReinitMap = pcl::getTime();
			reinitOccupancyMap();
			isStartPlaneRight = true;
			// std::cout << "Reinit occupancy map use " << pcl::getTime() - startReinitMap << " s" << std::endl;

			// double startShowCloud = pcl::getTime();
			//update the point in the viewer
			viewer.removeAllPointClouds ();
			viewer.removeAllShapes();
			pcl::visualization::PointCloudColorHandlerRGBField <PointT> rgb (cloud);
			pcl::visualization::PointCloudColorHandlerCustom<PointT> obstacleRed (cloud_obstacle, 255, 0, 0);
			pcl::visualization::PointCloudColorHandlerCustom<PointT> planeGreen (cloud_plane, 0, 255, 0);
			viewer.addPointCloud <PointT> (cloud, rgb, "new cloud");
			// viewer.addPointCloud <PointT> (cloud, rgb, "rgb cloud", v2);
			viewer.addPointCloud <PointT> (cloud_obstacle, obstacleRed, "obstacle_cloud");
			viewer.addPointCloud <PointT> (cloud_plane, planeGreen, "plane_cloud");
			// std::cout << "Show cloud use " << pcl::getTime() - startShowCloud << std::endl;
			viewer.spinOnce();

			// calculate the fps
			if ( ++count == 10)
			{
				double now = pcl::getTime();
				std::cout << "FPS = " << double (count) / double (now - last) << std::endl;
				count = 0;
				last = now;
				//Eigen::VectorXf &optimized_coefficients;
				calibratePlaneCoefficients ( cloud, inliers_plane, model_coefficients);
				setPlaneNormalVector(model_coefficients, plane_normalVector);
				// optimizeModelCoefficients ( inliers, model_coefficients, model_coefficients);
			}
			cloud_mutex.unlock ();
			// std::cout << "Time used = " << pcl::getTime() - time << std::endl;
			// std::cout << "One frame use " << pcl::getTime() - startFrame << " m" << std::endl;
		}
	}
	sendVelocity(0, 0);
	return 0;
}
