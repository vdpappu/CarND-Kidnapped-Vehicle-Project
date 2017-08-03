/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {


	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;
	normal_distribution<double> dist_x(0,std[0]);
	normal_distribution<double> dist_y(0,std[0]);
	normal_distribution<double> dist_theta(0,std[0]);

	for(int i=0;i<num_particles;++i)
	{
		Particle p;
		p.id = i;
		p.x = x + dist_x(gen);
		p.y = y + dist_y(gen);
		p.theta = theta + dist_theta(gen);
		p.weight = 1;

		weights.push_back(p.weight);
		particles.push_back(p);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	normal_distribution<double> noise_x(0,std_pos[0]);
	normal_distribution<double> noise_y(0,std_pos[1]);
	normal_distribution<double> noise_theta(0,std_pos[2]);

	for(int i=0;i<particles.size();++i)
	{
		double x_pred;
		double y_pred;
		double theta_pred;

		if(fabs(yaw_rate)<0.00001)
		{
			x_pred = particles[i].x + velocity * cos(particles[i].theta) * delta_t;
      		y_pred = particles[i].y + velocity * sin(particles[i].theta) * delta_t;
      		theta_pred = particles[i].theta + yaw_rate * delta_t;
		}
		else
		{
			x_pred = particles[i].x + (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      		y_pred = particles[i].y + (velocity / yaw_rate) * (-cos(particles[i].theta + yaw_rate * delta_t) + cos(particles[i].theta));
      		theta_pred = particles[i].theta + yaw_rate * delta_t;
		}
		particles[i].x = x_pred + noise_x(gen);
		particles[i].y = y_pred + noise_y(gen);
		particles[i].theta = theta_pred+noise_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i=0; i<observations.size(); ++i)
	{
		double t_distance = 100000;
		int nearest_landmark = -1;

		for(int j=0; j<=predicted.size();++j)
		{
			double dist_eucl = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
			if(dist_eucl<t_distance)
			{
				t_distance = dist_eucl;
				nearest_landmark = j;
			}
		}
		observations[i].id = predicted[nearest_landmark].id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) 
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  weights.clear();
	// Iterate through all particles
  for (unsigned i = 0; i < particles.size(); ++i) 
  {
	// Transform observation coordinates from vehicle to map
	std::vector<LandmarkObs> obs_map;
	for (unsigned int j = 0; j < observations.size(); ++j)
	{
		if (dist(observations[j].x, observations[j].y, 0, 0) <= sensor_range)
		{
			LandmarkObs obs;
			obs.x = particles[i].x + observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta);
			obs.y = particles[i].y + observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta);
			obs.id = -1;
			obs_map.push_back(obs);
		}
	}
	// Create a list of nearest landmarks in map coordinates
	std::vector<LandmarkObs> nearest_landmarks;
	for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j)
	{
		if (dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) <= sensor_range) {
			LandmarkObs obs;
			obs.x = map_landmarks.landmark_list[j].x_f;
			obs.y = map_landmarks.landmark_list[j].y_f;
			obs.id = map_landmarks.landmark_list[j].id_i;
			nearest_landmarks.push_back(obs);
		}
	}
	// Find the nearest landmark id for each observaton
	dataAssociation(nearest_landmarks, obs_map);

	// Calculate weights by factoring in multivariate Gaussian probabilities
	double weight = 1;
	for (unsigned int j = 0; j < nearest_landmarks.size(); j++) 
	{
		double dist_min = 1e6;
		int min_k = -1;
		// Iterate through map coordinates
		for (unsigned int k = 0; k < obs_map.size(); ++k) 
		{
			// Iterate through nearest landmark observations to find minimum distance
			if (obs_map[k].id == nearest_landmarks[j].id) 
			{
				double dist_eucl = dist(nearest_landmarks[j].x, nearest_landmarks[j].y, obs_map[k].x, obs_map[k].y);
				if (dist_eucl < dist_min) 
				{
					dist_min = dist_eucl;
					min_k = k;
				}
			}
		}
		if (min_k != -1) 
		{
			// Create variables for clarity
			double x = obs_map[min_k].x;
			double y = obs_map[min_k].y;
			double mu_x = nearest_landmarks[j].x;
			double mu_y = nearest_landmarks[j].y;
			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];
			// Calculate weight using multivariate Gaussian equation
    		weight *= exp(-((x - mu_x) * (x - mu_x) / (2 * sig_x * sig_x) + (y - mu_y) * (y - mu_y) / (2 * sig_y * sig_y))) / (2 * M_PI * sig_x * sig_y);
  		}
	}
	// Update particles with correct weights
	weights.push_back(weight);
	particles[i].weight = weight;

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles;
	vector<double> weights;

	for (int i = 0; i < num_particles; i++)
	{
    	weights.push_back(particles[i].weight);
  	}
  	uniform_int_distribution<int> uniintdist(0, num_particles-1);
  	auto index = uniintdist(gen);
  	double max_weight = *max_element(weights.begin(), weights.end());
  	uniform_real_distribution<double> unirealdist(0.0, max_weight);
  	double beta = 0.0;
  	//resampling
	for (int i = 0; i < num_particles; i++)
	{
	    beta += unirealdist(gen) * 2.0;
	    while (beta > weights[index])
	    {
	      beta -= weights[index];
	      index = (index + 1) % num_particles;
	    }
	    new_particles.push_back(particles[index]);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
