import React, { useState } from 'react';
import { 
    Navbar as NavigationBar,
    Button
} from 'react-bootstrap';

import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';

const Navbar = () => {
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { signout } = useAuth();
    const navigate = useNavigate();

    const handleSignout = async (e) => {
        e.preventDefault();

        try {
            setError('');
            setLoading(true);
            await signout();
            navigate('/');
        } catch (error) {
            setError('Failed to sign out');
            setLoading(false);
        }
    }

    return (
        <div>
            <NavigationBar bg="light" className="justify-content-between">
                    
                <NavigationBar.Brand style={{ marginLeft: '15px' }}>Dashboard</NavigationBar.Brand>
                <Button style={{ marginRight: '24px' }} variant="outline-danger" onClick={handleSignout}>Log Out</Button>
            </NavigationBar>
        </div>
    );
}

export default Navbar;
