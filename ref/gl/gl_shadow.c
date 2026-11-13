/*
 * Xash3D-FWGS shadow mapping.
 *
 * Desktop compatibility OpenGL uses a depth texture/FBO and the third
 * fixed-function texture unit for hardware depth comparison. Other backends
 * keep rendering without shadows rather than entering a partial GL path.
 */
#include "gl_local.h"
#include "xash3d_mathlib.h"

CVAR_DEFINE_AUTO( r_shadows, "0", FCVAR_GLCONFIG, "enable dynamic shadow mapping" );
CVAR_DEFINE_AUTO( r_shadowmap_size, "1024", FCVAR_GLCONFIG, "shadow map resolution (256 to 4096)" );
CVAR_DEFINE_AUTO( r_shadowmap_distance, "2048", FCVAR_GLCONFIG, "shadow map radius around the view" );
CVAR_DEFINE_AUTO( r_shadowmap_bias, "0.0015", FCVAR_GLCONFIG, "shadow map depth bias" );
CVAR_DEFINE_AUTO( r_shadowmap_lightdir_x, "-0.45", FCVAR_GLCONFIG, "shadow light direction X" );
CVAR_DEFINE_AUTO( r_shadowmap_lightdir_y, "-0.35", FCVAR_GLCONFIG, "shadow light direction Y" );
CVAR_DEFINE_AUTO( r_shadowmap_lightdir_z, "-1.0", FCVAR_GLCONFIG, "shadow light direction Z" );

#if !XASH_GLES
typedef struct
{
	GLuint fbo, depthTexture;
	int size, failedSize;
	qboolean valid, rendering, receiving, warned, failed;
	GLint oldFbo, oldViewport[4];
	matrix4x4 lightViewProjection;
} shadowmap_t;

static shadowmap_t g_shadowmap;

static void R_ShadowMatrixMultiply( matrix4x4 out, const matrix4x4 a, const matrix4x4 b )
{
	matrix4x4 result;
	for( int row = 0; row < 4; row++ )
		for( int col = 0; col < 4; col++ )
		{
			result[row][col] = 0.0f;
			for( int k = 0; k < 4; k++ )
				result[row][col] += a[row][k] * b[k][col];
		}
	Matrix4x4_Copy( out, result );
}

static void R_ShadowMatrixOrtho( matrix4x4 m, float radius, float zNear, float zFar )
{
	Matrix4x4_LoadIdentity( m );
	m[0][0] = m[1][1] = 1.0f / radius;
	m[2][2] = -2.0f / ( zFar - zNear );
	m[2][3] = -( zFar + zNear ) / ( zFar - zNear );
}

static qboolean R_ShadowMapSupported( void )
{
	qboolean shadowCompare = ( glConfig.version_major > 1 ||
		( glConfig.version_major == 1 && glConfig.version_minor >= 4 ) ||
		Q_stristr( glConfig.extensions_string, "GL_ARB_shadow" ) != NULL );

	if( glConfig.context == CONTEXT_TYPE_GL_CORE )
		return false;

	return GL_Support( GL_FRAMEBUFFER_OBJECT_EXT ) &&
		GL_Support( GL_DEPTH_TEXTURE ) &&
		GL_Support( GL_ARB_MULTITEXTURE ) &&
		glConfig.max_texture_units >= 3 &&
		glConfig.max_texture_coords >= 3 && shadowCompare;
}

static void R_ShadowMapDestroy( void )
{
	if( g_shadowmap.fbo )
		pglDeleteFramebuffers( 1, &g_shadowmap.fbo );
	if( g_shadowmap.depthTexture )
		pglDeleteTextures( 1, &g_shadowmap.depthTexture );

	/* Raw texture binds bypass GL_Bind's cache. */
	glState.currentTextures[0] = glState.currentTextures[2] = -1;
	glState.currentTexturesIndex[0] = glState.currentTexturesIndex[2] = 0;
	memset( &g_shadowmap, 0, sizeof( g_shadowmap ));
}

static int R_ShadowMapResolution( void )
{
	int requested = bound( 256, (int)r_shadowmap_size.value, 4096 );
	int maximum = Q_min( glConfig.max_2d_texture_size, 4096 );
	int size = 1;

	if( maximum < 1 )
		return 0;
	while( size < requested && size <= maximum / 2 )
		size <<= 1;
	if( size > maximum )
		size = maximum;
	return size;
}

static qboolean R_ShadowMapCreate( void )
{
	GLint oldFbo = 0;
	int size = R_ShadowMapResolution();
	int oldTMU = glState.activeTMU;
	GLenum status;
	const float border[4] = { 1, 1, 1, 1 };

	if( size <= 0 )
		return false;
	if( g_shadowmap.fbo && g_shadowmap.depthTexture && g_shadowmap.size == size )
		return true;
	if( g_shadowmap.failed && g_shadowmap.failedSize == size )
		return false;

	R_ShadowMapDestroy();
	pglGetIntegerv( GL_FRAMEBUFFER_BINDING, &oldFbo );
	GL_SelectTexture( 0 );
	pglGenTextures( 1, &g_shadowmap.depthTexture );
	pglBindTexture( GL_TEXTURE_2D, g_shadowmap.depthTexture );
	glState.currentTextures[0] = -1;
	glState.currentTexturesIndex[0] = 0;

	pglTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, size, size, 0,
		GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, NULL );
	pglTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	pglTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	pglTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB, GL_COMPARE_R_TO_TEXTURE_ARB );
	pglTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC_ARB, GL_LEQUAL );
	pglTexParameteri( GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE_ARB, GL_INTENSITY );

	if( GL_Support( GL_CLAMP_TEXBORDER_EXT ))
	{
		pglTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
		pglTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
		pglTexParameterfv( GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border );
	}
	else
	{
		pglTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
		pglTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	}

	pglGenFramebuffers( 1, &g_shadowmap.fbo );
	pglBindFramebuffer( GL_FRAMEBUFFER, g_shadowmap.fbo );
	pglFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, g_shadowmap.depthTexture, 0 );
	pglDrawBuffer( GL_NONE );
	pglReadBuffer( GL_NONE );
	status = pglCheckFramebufferStatus( GL_FRAMEBUFFER );
	pglBindFramebuffer( GL_FRAMEBUFFER, oldFbo );
	GL_SelectTexture( oldTMU );

	if( status != GL_FRAMEBUFFER_COMPLETE )
	{
		gEngfuncs.Con_Printf( S_WARN "r_shadows: depth framebuffer is incomplete (0x%x); shadows disabled\n", status );
		R_ShadowMapDestroy();
		g_shadowmap.failed = true;
		g_shadowmap.failedSize = size;
		return false;
	}

	g_shadowmap.size = size;
	gEngfuncs.Con_DPrintf( "r_shadows: created %dx%d depth map\n", size, size );
	return true;
}

qboolean R_ShadowMapBegin( void )
{
	vec3_t lightDirection, lightAngles, lightOrigin;
	float radius, zFar;

	if( !r_shadows.value )
		return false;
	if( !R_ShadowMapSupported() )
	{
		if( !g_shadowmap.warned )
		{
			gEngfuncs.Con_Printf( S_WARN "r_shadows requires desktop compatibility GL, depth textures, FBOs and at least three texture units\n" );
			g_shadowmap.warned = true;
		}
		return false;
	}
	if( !R_ShadowMapCreate() )
		return false;

	g_shadowmap.valid = false;
	pglGetIntegerv( GL_FRAMEBUFFER_BINDING, &g_shadowmap.oldFbo );
	pglGetIntegerv( GL_VIEWPORT, g_shadowmap.oldViewport );

	radius = bound( 256.0f, r_shadowmap_distance.value, 8192.0f );
	VectorSet( lightDirection, r_shadowmap_lightdir_x.value, r_shadowmap_lightdir_y.value, r_shadowmap_lightdir_z.value );
	if( DotProduct( lightDirection, lightDirection ) < 0.0001f )
		VectorSet( lightDirection, -0.45f, -0.35f, -1.0f );
	VectorNormalize( lightDirection );
	VectorMA( RI.rvp.vieworigin, -radius, lightDirection, lightOrigin );
	VectorAngles( lightDirection, lightAngles );

	VectorCopy( lightOrigin, RI.rvp.vieworigin );
	VectorCopy( lightAngles, RI.rvp.viewangles );
	RI.rvp.viewport[0] = RI.rvp.viewport[1] = 0;
	RI.rvp.viewport[2] = RI.rvp.viewport[3] = g_shadowmap.size;
	RI.rvp.fov_x = RI.rvp.fov_y = 90.0f;

	R_SetupFrustum();
	/* A locked camera frustum must not lock the light-space cull origin. */
	VectorCopy( RI.rvp.vieworigin, RI.cullorigin );
	VectorCopy( RI.vforward, RI.cull_vforward );
	VectorCopy( RI.vright, RI.cull_vright );
	VectorCopy( RI.vup, RI.cull_vup );
	RI.viewplanedist = DotProduct( RI.rvp.vieworigin, RI.vforward );
	R_FindViewLeaf();
	R_SetupGL( false );

	zFar = Q_max( radius * 4.0f, RI.farClip );
	R_ShadowMatrixOrtho( RI.projectionMatrix, radius, 1.0f, zFar );
	R_ShadowMatrixMultiply( RI.worldviewProjectionMatrix, RI.projectionMatrix, RI.worldviewMatrix );
	Matrix4x4_Copy( g_shadowmap.lightViewProjection, RI.worldviewProjectionMatrix );
	GL_FrustumInitOrtho( &RI.frustum, -radius, radius, radius, -radius, 1.0f, zFar );

	pglBindFramebuffer( GL_FRAMEBUFFER, g_shadowmap.fbo );
	pglDrawBuffer( GL_NONE );
	pglReadBuffer( GL_NONE );
	pglViewport( 0, 0, g_shadowmap.size, g_shadowmap.size );
	pglEnable( GL_DEPTH_TEST );
	pglDepthFunc( GL_LEQUAL );
	pglDepthMask( GL_TRUE );
	pglColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );
	pglDisable( GL_BLEND );
	pglDisable( GL_FOG );
	pglEnable( GL_POLYGON_OFFSET_FILL );
	pglPolygonOffset( 2.0f, 4.0f );
	pglClearDepth( 1.0 );
	pglClear( GL_DEPTH_BUFFER_BIT );

	pglMatrixMode( GL_PROJECTION );
	GL_LoadMatrix( RI.projectionMatrix );
	pglMatrixMode( GL_MODELVIEW );
	GL_LoadMatrix( RI.worldviewMatrix );
	GL_Cull( GL_FRONT );

	g_shadowmap.rendering = true;
	return true;
}

void R_ShadowMapEnd( void )
{
	if( !g_shadowmap.rendering )
		return;

	g_shadowmap.rendering = false;
	pglColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
	pglDisable( GL_POLYGON_OFFSET_FILL );
	pglDepthMask( GL_TRUE );
	pglDepthFunc( GL_LEQUAL );
	pglDisable( GL_BLEND );
	pglBindFramebuffer( GL_FRAMEBUFFER, g_shadowmap.oldFbo );
	pglViewport( g_shadowmap.oldViewport[0], g_shadowmap.oldViewport[1],
		g_shadowmap.oldViewport[2], g_shadowmap.oldViewport[3] );

	GL_CleanupAllTextureUnits();
	GL_SelectTexture( 0 );
	g_shadowmap.valid = true;
}

void R_ShadowMapEnable( void )
{
	matrix4x4 inverseView, projected, bias, textureMatrix;
	const vec4_t planeS = { 1, 0, 0, 0 };
	const vec4_t planeT = { 0, 1, 0, 0 };
	const vec4_t planeR = { 0, 0, 1, 0 };
	const vec4_t planeQ = { 0, 0, 0, 1 };

	if( !r_shadows.value || !g_shadowmap.valid || g_shadowmap.receiving )
		return;
	if( !Matrix4x4_Invert_Full( inverseView, RI.worldviewMatrix ))
		return;

	Matrix4x4_LoadIdentity( bias );
	bias[0][0] = bias[1][1] = bias[2][2] = 0.5f;
	bias[0][3] = bias[1][3] = 0.5f;
	bias[2][3] = 0.5f - bound( 0.0f, r_shadowmap_bias.value, 0.02f );
	R_ShadowMatrixMultiply( projected, g_shadowmap.lightViewProjection, inverseView );
	R_ShadowMatrixMultiply( textureMatrix, bias, projected );

	GL_SelectTexture( 2 );
	GL_TextureTarget( GL_TEXTURE_2D );
	pglBindTexture( GL_TEXTURE_2D, g_shadowmap.depthTexture );
	glState.currentTextures[2] = -1;
	glState.currentTexturesIndex[2] = 0;
	pglTexEnvi( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );

	/* Store eye-linear planes in eye space, independent of the current view. */
	pglMatrixMode( GL_MODELVIEW );
	pglPushMatrix();
	pglLoadIdentity();
	GL_TexGen( GL_S, GL_EYE_LINEAR );
	GL_TexGen( GL_T, GL_EYE_LINEAR );
	GL_TexGen( GL_R, GL_EYE_LINEAR );
	GL_TexGen( GL_Q, GL_EYE_LINEAR );
	pglTexGenfv( GL_S, GL_EYE_PLANE, planeS );
	pglTexGenfv( GL_T, GL_EYE_PLANE, planeT );
	pglTexGenfv( GL_R, GL_EYE_PLANE, planeR );
	pglTexGenfv( GL_Q, GL_EYE_PLANE, planeQ );
	pglPopMatrix();

	pglMatrixMode( GL_TEXTURE );
	GL_LoadMatrix( textureMatrix );
	glState.texIdentityMatrix[2] = false;
	pglMatrixMode( GL_MODELVIEW );
	GL_SelectTexture( 0 );
	g_shadowmap.receiving = true;
}

void R_ShadowMapDisable( void )
{
	if( !g_shadowmap.receiving )
		return;

	GL_SelectTexture( 2 );
	GL_TexGen( GL_S, 0 );
	GL_TexGen( GL_T, 0 );
	GL_TexGen( GL_R, 0 );
	GL_TexGen( GL_Q, 0 );
	GL_LoadIdentityTexMatrix();
	GL_TextureTarget( GL_NONE );
	glState.currentTextures[2] = -1;
	glState.currentTexturesIndex[2] = 0;
	GL_SelectTexture( 0 );
	g_shadowmap.receiving = false;
}

qboolean R_ShadowMapIsRendering( void )
{
	return g_shadowmap.rendering;
}

#else /* XASH_GLES */
qboolean R_ShadowMapBegin( void ) { return false; }
void R_ShadowMapEnd( void ) {}
void R_ShadowMapEnable( void ) {}
void R_ShadowMapDisable( void ) {}
qboolean R_ShadowMapIsRendering( void ) { return false; }
#endif /* !XASH_GLES */

void R_ShadowMapInit( void )
{
	gEngfuncs.Cvar_RegisterVariable( &r_shadowmap_size );
	gEngfuncs.Cvar_RegisterVariable( &r_shadowmap_distance );
	gEngfuncs.Cvar_RegisterVariable( &r_shadowmap_bias );
	gEngfuncs.Cvar_RegisterVariable( &r_shadowmap_lightdir_x );
	gEngfuncs.Cvar_RegisterVariable( &r_shadowmap_lightdir_y );
	gEngfuncs.Cvar_RegisterVariable( &r_shadowmap_lightdir_z );
}

void R_ShadowMapShutdown( void )
{
#if !XASH_GLES
	R_ShadowMapDisable();
	R_ShadowMapDestroy();
#endif
}
