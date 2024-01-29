/**
* Name: lenia
* Based on LENIA implementation https://arxiv.org/pdf/1812.05433.pdf 
* Author: kevinchapuis
* Tags: Artificial life, game of life, lenia
*/


model lenia

global {
	
	int L <- 30;
	int maxval <- 255;
	
	geometry shape <- square(L);
	field f <- field(L,L,0);
	
	float DT <- 1.0;
	float MU <- 0.14;
	float SIGMA <- 0.014;
	bool KS <- true;
	
	field __fieldK;
	
	field __fieldGPotential;
	
	init {
		
		//RED
		matrix m <- {f.columns,f.rows} matrix_with 1.0;
		loop x from:0 to:f.columns-1 { loop y from:0 to:f.rows-1 { m[{x,y}] <- rnd(maxval); } }
		create lenia with:[space::field(m)];
		f.bands <+ last(lenia).space;
		
		//GREEN
//		matrix m1 <- {f.columns,f.rows} matrix_with 1.0;
//		loop x from:0 to:f.columns-1 { loop y from:0 to:f.rows-1 { m1[{x,y}] <- rnd(maxval); } }
//		create lenia with:[space::field(m1)];
//		f.bands <+ last(lenia).space;
		
		//BLUE
//		matrix m2 <- {f.columns,f.rows} matrix_with 1.0;
//		loop x from:0 to:f.columns-1 { loop y from:0 to:f.rows-1 { m2[{x,y}] <- rnd(maxval); } }
//		create lenia with:[space::field(m2)];
//		f.bands <+ last(lenia).space;
		
		// PRE COMPUTE FILTERS
		create kernel with:[mask::KS?compute_filter():compute_filter(beta::[1])];
		__fieldK <- first(kernel).from_mask_to_field({L/2,L/2});
		
		lenia(0).filters <- lenia as_map (each::first(kernel));
		
		// CONVOLVE KERNEL AT INIT
		map<lenia,map<lenia,matrix>> am;
		ask lenia parallel:true { am[self] <- activation_matrix() ; }
		__fieldGPotential <- field(first(am.values)[first(lenia)]);
		
	}
	
	//********//
	// KERNEL //
	//********//
	
	float EPSILON <- 10^-6;
	
	int R <- 15;
	list<float> BETA <- [1,2/3,1/3];
	
	float ALPHA <- 4;
	
	map<point,float> compute_filter(int range <- R, list<float> beta <- BETA) {
		map<point,float> res;
		loop x from:0-range to:0+range {
			loop y from:0-range to:0+range {
				// TODO replace relative euclidian distance by polar distance (is it necessary ?)
				float ed <- sqrt( x*x + y*y );
				if (ed >= range) { continue; } // TODO : should be Moor distance, but here is a square around {x,y} of size {R,R}
				
				// find beta shell
				float rank <- ed / range * length(beta);
				// to avoid 0 distance
				float Br <- min(rank * ed / range - int(rank * ed / range) + EPSILON, 1.0);
				
				// Actual kernle core & shell
				float K <- kfunc(Br, beta[rank]);
				if K > 0 { res[{x,y}] <- K; }
			}
		}
		return res;
	}
	
	// ******************************************************** //
	
	/*
	 * 
	 */
	reflex evolve {
		map<lenia,map<lenia,matrix>> am;
		ask lenia parallel:true { am[self] <- activation_matrix() ; }
		__fieldGPotential <- field(first(am.values)[first(lenia)]);
		ask lenia parallel:true { do evolve(am[self]); f.bands[int(self)] <- space; }
	}
	
	/*
	 * TODO implement polynomial
	 */
	action kfunc(float r, float b, float a <- ALPHA) { return b * exp( a - (a / (4*r*(1-r))) ); }
	
}

species lenia { 
	
	map<lenia, kernel> filters;
	
	field space;
	
	map<lenia,matrix> activation_matrix {
		
		map<lenia,matrix> me <- filters.keys as_map (each::({space.columns,space.rows} matrix_with 0.0));
		// Compute activation
		loop x from:0 to:space.columns-1 {
			loop y from:0 to:space.rows-1 {
				loop l over:filters.keys {
					field f <- filters[l].from_mask_to_field({x,y});
					float u <- sum(f / (sum(f)*1/R^2) * (space[{x,y}]/maxval + copy(l.space)/maxval)) * 1/R^2;
					write u;
					me[l][{x,y}] <- activation(u);
				}
				
			}
		}
		return me;
	}
	
	action evolve(map<lenia,matrix> am) {
		// Apply activation
		loop x from:0 to:space.columns-1 {
			loop y from:0 to:space.rows-1 {
				float a <- mean(am.values collect float(each[{x,y}]));
				// scale out results to feet RGB on 3 dimensional Lenia
				// TODO : may be implement soft clip
				space[{x,y}] <- max(0, min(maxval, space[{x,y}] + space[{x,y}] * a * DT));	
			}
		}
		
	}
	
	float activation(float u) { return 2 * exp( - ((u-MU) ^ 2) / (2 * SIGMA ^ 2)) - 1; }
	
}

species kernel { 
	map<point,float> mask;
	
	field from_mask_to_field(point cell) {
		field res <- field(L,L,0.0);
		loop mp over:mask.keys { res[point_to_boundary(cell+mp)] <- mask[mp]; }
		return res;
	}
	
	point point_to_boundary(point cell) {
		// deal with borders
		int bx <- cell.x;
		if bx < 0 { bx <- L + bx; }
		if bx > L - 1 { bx <- bx - L; }
		// deal with borders
		int by <- cell.y;
		if by < 0 { by <- L + by; }
		if by > L - 1 { by <- by - L; }
		return {bx,by};
	}
}

experiment xp {
	
	parameter delta var:DT min:0.01 max:1.0;
	parameter mu var:MU min:0.1 max:0.5;
	parameter sigma var:SIGMA min:0.01 max:0.11;
	parameter shell var:KS init:false;
	
	output {
		display main {
			//mesh f color:f.bands scale:0 triangulation:false smooth:false;
			mesh f.bands[0] grayscale:true scale:0;
			graphics v {
				loop x from:0 to:f.columns {
					loop y from:0 to:f.rows {
						draw string(with_precision(f.bands[0][{x,y}],1)) font:font("Arial",5,5) anchor:#top_left color:#darkred at:{x,y};
					}
				}
			}
		}
		display potential {
			mesh __fieldGPotential color:palette([#red,#white,#green]) scale:0 refresh:true;
		}
		display kernel {
			mesh __fieldK grayscale:true scale:0 triangulation:false smooth:false;
		}
	}
}