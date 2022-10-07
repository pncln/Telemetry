import React, { useState, useEffect } from 'react';

import Navbar from './dashboard/Navbar';
import Home from './dashboard/Home';
import Settings from './dashboard/Settings';


import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';

import { 
    Col,
    Tab,
    ListGroup,
    Row
} from 'react-bootstrap';

import './dashboard/dash-styles.css';

const DashScreen = () => {
    const [key, setKey] = useState("#home");
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();
    const { currentUser } = useAuth();

    useEffect(() => {
        const handleNotLoggedIn = async (e) => {
            if (!currentUser) {
                try {
                    setError('');
                    setLoading(true);
                    navigate('/');
                } catch (error) {
                    console.log(error);
                }
            }
        };

        handleNotLoggedIn();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    const Page = () => (
        <div>
            <Navbar />
            <Tab.Container
                id="tabs" 
                activeKey={key}
                onSelect={(k) => setKey(k)}>
                <Row>
                    <Col sm={3}>
                    <ListGroup variant="flush">
                        <ListGroup.Item action href="#home">
                        Home
                        </ListGroup.Item>
                        <ListGroup.Item action href="#settings">
                        Settings
                        </ListGroup.Item>
                    </ListGroup>
                    </Col>
                    <Col sm={9}>
                    <Tab.Content style={{ marginRight: 25 }}>
                        <Tab.Pane eventKey="#home">
                            <Home />
                        </Tab.Pane>
                        <Tab.Pane eventKey="#settings">
                            <Settings />
                        </Tab.Pane>
                    </Tab.Content>
                    </Col>
                </Row>
            </Tab.Container>
        </div>
    )

    return (
        currentUser ? <Page /> : null
    );
};

export default DashScreen;
