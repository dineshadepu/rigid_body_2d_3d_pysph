### Setting the velocity of the rigid body
	The velocity of the rigid body can be set in the following way
	```python
	self.scheme.scheme.set_linear_velocity(body1, np.array([0.5, 0., 0.]))
	```
	if the body has many sub bodies, identified separately with body id's then

	```python
	self.scheme.scheme.set_linear_velocity(body1, np.array([0.5, 0., 0., 0.5., 0., 0.]))
	```

	DANGER. If you set it with a single element, it leads to the segfault.
