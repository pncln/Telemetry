import React, { useRef, useState, useEffect } from 'react';
import {
    Form,
    Button,
    Card,
    Alert,
    Container
} from 'react-bootstrap';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

const Signin = () => {
    const emailRef = useRef();
    const passwordRef = useRef();
    const { signin } = useAuth();
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();
    const { currentUser } = useAuth();

    useEffect(() => {
        const handleLoggedIn = async (e) => {
            if (currentUser) {
                try {
                    setError('');
                    setLoading(true);
                    navigate('/dashboard');
                } catch (error) {
                    console.log(error);
                }
            }
        };

        handleLoggedIn();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    async function handleSubmit(e) {
        e.preventDefault();

        try {
            setError('');
            setLoading(true);
            await signin(emailRef.current.value, passwordRef.current.value);
            navigate('/dashboard');
        } catch (e) {
            console.log(e);
            setError('Failed to sign in!');
            setLoading(false);
        }
    }

    return (
        <div className="background-image-container">
            <Container style={{ height: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <Card style={{ margin: '100px', width: '400px', maxHeight: '90%' }}>
                    <Card.Body style={{ margin: '10px' }}>
                        <div>
                            {error && <Alert variant="danger"> {error} </Alert> }
                        <div className="col-title">
                            Log In
                        </div>
                        <br />
                        <Form onSubmit={handleSubmit}>
                            <Form.Group id="email">
                                <Form.Label>Email address</Form.Label>
                                <Form.Control type="email" ref={emailRef}  placeholder="Enter email" required />
                            </Form.Group>

                            <Form.Group id="password">
                                <Form.Label>Password</Form.Label>
                                <Form.Control type="password" ref={passwordRef} placeholder="Password" required />
                            </Form.Group>
                            <div className="auth-form-bottom">
                                <Button style={{ paddingLeft: '30px', paddingRight: '35px' }} disabled={loading} variant="primary" type="submit">
                                    Log In
                                </Button>
                            </div>
                        </Form>
                        </div>
                    </Card.Body>
                </Card>
            </Container>
        </div>
    )
};

export default Signin;