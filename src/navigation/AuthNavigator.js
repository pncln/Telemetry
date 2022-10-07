import React, { useEffect, useContext, useState } from 'react';
import SigninScreen from '../screens/SigninScreen';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext'

export const AuthNavigator = () => {
    const { currentUser } = useAuth();
    const [ error, setError ] = useState('');
    const navigate = useNavigate();

    return (
        <SigninScreen />
    )
}